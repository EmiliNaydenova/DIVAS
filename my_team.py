# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
"""
Team DIVAS: one attacker and one defender.

- Both agents share a belief (probability over cells) for each opponent,
  updated every turn from the noisy distances, from what we can and cannot
  see, from which half the opponent is on and from food that disappears.
- The attacker computes for every enemy cell the earliest time a ghost could
  be there and only takes routes on which it arrives first.  If nothing is
  safe and it carries little, it takes the least risky route instead.
- At start-up we solve the one-ghost-vs-one-Pacman game exactly (minimax by
  backward induction).  The defender uses it to chase invaders, the attacker
  uses it to avoid walking into a forced capture.
- If an agent keeps revisiting the same few cells without eating, banking or
  dying, it commits to a new target and path for a few turns.
"""

import heapq
import random
import time
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions


def create_team(first_index, second_index, is_red,
                first='AttackAgent', second='DefendAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


INF = 10 ** 6

# Game facts
SIGHT = 5            # we see enemies within this Manhattan distance
SONAR_NOISE = 6      # noisy distances are off by at most this much

# Attack and banking
CARRY_LIMIT = 5      # go home with this much food (unless nobody defends)
RISK_CARRY = 1       # take risky routes only while carrying at most this
RISK_PENALTY = 4     # extra cost per move we would lose the race to a ghost
POCKET_PENALTY = 3   # same, for getting back out of a dead end
UNSEEN_SLACK = 2     # unseen ghosts are assumed this many moves further away

# Team coordination
HELP_CARRY = 6       # attacker comes back when invaders carry this much
HELP_MAX_TURNS = 20  # ...but stops helping after this many turns
DOUBLE_AFTER = 6     # attacker blocked this long: defender joins the attack
GUARD_PATIENCE = 6   # turns face to face with a ghost before backing off
BAIT_TURNS = 10      # how long we back off
BAIT_DIST = 3        # how far we back off

# Recovery (getting unstuck)
STALL_TURNS = 4      # turns without progress before we try something new
STALL_WINDOW = 8     # a cell visited this recently counts as a revisit
COMMIT_TURNS = 10    # how long we stick to the new plan

# Duel limits
DUEL_DEPTH = 8       # forced captures further away than this are ignored
DUEL_PATIENCE = 6    # stop avoiding duels after circling this many turns
BUILD_BUDGET = 6.0   # seconds of start-up spent solving duels at most


_TEAM_DATA = {}      # shared data of each team, keyed by is_red


class Duel:
    """One ghost against one Pacman on one half of the map, solved exactly.

    A state is (ghost cell, Pacman cell, whose turn).  The ghost wins by
    catching the Pacman, the Pacman wins by leaving the half or reaching a
    capsule.  We start from the finished states and work backwards (like
    value iteration), so every state gets one of:
        GHOST  the ghost can force a capture in `depth` moves
        PAC    the Pacman can force an escape in `depth` moves
        DRAW   neither (the ghost can keep the Pacman in, but not catch it)
    Both players may also stand still."""
    DRAW, GHOST, PAC = 0, 1, 2

    def __init__(self, td, in_half, capsules, deadline):
        cells = [cell for cell in td.cells if in_half(cell)]
        index_of = {cell: i for i, cell in enumerate(cells)}
        n = len(cells)
        self.index_of, self.n, self.ok = index_of, n, False
        # moves[i]: cells reachable from cell i in one move (staying put included)
        moves = [[index_of[nbr] for _, nbr in td.nbrs[cell] if nbr in index_of] + [i]
                 for i, cell in enumerate(cells)]
        can_leave = [any(nbr not in index_of for _, nbr in td.nbrs[cell]) for cell in cells]

        # Each state is packed into one integer:
        #     state_id = (ghost_index * n + pacman_index) * 2 + turn
        # where turn is 0 when the ghost moves next and 1 when the Pacman does.
        outcomes = bytearray(2 * n * n)
        depths = [0] * (2 * n * n)
        open_moves = [0] * (2 * n * n)   # moves not yet known to lose
        queue, one_step = [], []
        for ghost_index in range(n):
            for pacman_index in range(n):
                state_id = (ghost_index * n + pacman_index) * 2
                if ghost_index == pacman_index:
                    outcomes[state_id] = outcomes[state_id + 1] = Duel.GHOST
                    queue += [state_id, state_id + 1]
                elif cells[pacman_index] in capsules:
                    outcomes[state_id] = outcomes[state_id + 1] = Duel.PAC
                    queue += [state_id, state_id + 1]
                else:
                    open_moves[state_id] = len(moves[ghost_index])
                    open_moves[state_id + 1] = len(moves[pacman_index])
                    if can_leave[pacman_index]:   # Pacman can step home right away
                        outcomes[state_id + 1], depths[state_id + 1] = Duel.PAC, 1
                        one_step.append(state_id + 1)
        queue += one_step

        head = 0
        while head < len(queue):
            if head % 4096 == 0 and time.perf_counter() > deadline:
                return                        # out of time: table not usable
            state_id = queue[head]
            head += 1
            outcome, depth = outcomes[state_id], depths[state_id] + 1
            ghost_index, pacman_index = divmod(state_id // 2, n)
            if state_id % 2 == 1:
                # Pacman's turn here, so the ghost moved before.
                mover_wins = Duel.GHOST
                earlier = [(g0 * n + pacman_index) * 2 for g0 in moves[ghost_index]]
            else:
                mover_wins = Duel.PAC
                earlier = [(ghost_index * n + p0) * 2 + 1 for p0 in moves[pacman_index]]
            for earlier_id in earlier:
                if outcomes[earlier_id]:
                    continue
                if outcome == mover_wins:     # one winning move is enough
                    outcomes[earlier_id], depths[earlier_id] = outcome, depth
                    queue.append(earlier_id)
                else:                         # loses only if every move loses
                    open_moves[earlier_id] -= 1
                    if open_moves[earlier_id] == 0:
                        outcomes[earlier_id], depths[earlier_id] = outcome, depth
                        queue.append(earlier_id)
        self.outcomes, self.depths, self.ok = outcomes, depths, True

    def get(self, ghost, pac, pac_turn):
        """(outcome, depth), or None if a cell is not on this half."""
        ghost_index, pacman_index = self.index_of.get(ghost), self.index_of.get(pac)
        if ghost_index is None or pacman_index is None:
            return None
        state_id = (ghost_index * self.n + pacman_index) * 2 + (1 if pac_turn else 0)
        return self.outcomes[state_id], self.depths[state_id]


class TeamData:
    """Map analysis, opponent beliefs and duel tables shared by both agents."""

    def __init__(self, agent, game_state):
        walls = game_state.get_walls()
        self.red = agent.red
        self.height = walls.height
        self.cells = [(x, y) for x in range(walls.width) for y in range(walls.height)
                      if not walls[x][y]]
        self.nbrs = {}
        for c in self.cells:
            self.nbrs[c] = []
            for a in (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST):
                dx, dy = Actions.direction_to_vector(a)
                n = (int(c[0] + dx), int(c[1] + dy))
                if not walls[n[0]][n[1]]:
                    self.nbrs[c].append((a, n))

        self.mid = walls.width // 2
        home_x = self.mid - 1 if self.red else self.mid
        enemy_x = self.mid if self.red else self.mid - 1
        self.home_border = [(home_x, y) for y in range(walls.height) if not walls[home_x][y]]
        self.crossings = [c for c in self.home_border if not walls[enemy_x][c[1]]]
        self.find_dead_ends()

        self.opponents = agent.get_opponents(game_state)
        self.agent_ids = agent.get_team(game_state)
        self.opp_start = {o: game_state.get_initial_agent_position(o) for o in self.opponents}
        self.beliefs = {o: {self.opp_start[o]: 1.0} for o in self.opponents}
        self.last_seen = {o: None for o in self.opponents}
        self.prev_food = set(agent.get_food_you_are_defending(game_state).as_list())
        self.prev_caps = set(agent.get_capsules_you_are_defending(game_state))
        self.last_update = None
        self.initial_time = game_state.data.timeleft
        self.registered = set()

        self.blocked_turns = 0       # how long the attacker has been stuck at the border
        self.attack_target = None
        self.doubling = False        # defender has joined the attack
        self.defend_target = {}      # agent index -> opponent it is chasing

        # Solve the duels now for every set of capsules that can still be
        # left (capsules only disappear), so no table is built during a move.
        self.duels = {}
        self.duel_cost = INF
        deadline = time.perf_counter() + BUILD_BUDGET
        jobs = []
        for side, caps in (('def', agent.get_capsules_you_are_defending(game_state)),
                           ('att', agent.get_capsules(game_state))):
            caps = sorted(caps)
            subsets = [frozenset(c for i, c in enumerate(caps) if mask >> i & 1)
                       for mask in range(1 << min(len(caps), 4))]
            subsets.append(frozenset(caps))
            for sub in set(subsets):
                order = 0 if len(sub) == len(caps) else 1 if not sub else 2
                jobs.append((order, side, sub))
        for _, side, sub in sorted(jobs, key=lambda j: j[0]):
            self.duel(side, sub, deadline)

    def duel(self, side, caps, deadline=None):
        """Duel table for our half ('def') or the enemy half ('att'), or None.

        Tables are normally built at start-up.  A set of capsules that was
        not built then (only possible with more than 4 capsules) is built
        here during play if a build is quick, otherwise we fall back."""
        key = (side, frozenset(caps))
        if key in self.duels:
            return self.duels[key]
        if deadline is None:
            # Not built at start-up (more than 4 capsules): only build it now
            # if that is quick, otherwise pretend the capsules are gone.
            if self.duel_cost > 0.02:
                return self.duels.get(('att', frozenset())) if side == 'att' else None
            deadline = time.perf_counter() + 0.08
        t0 = time.perf_counter()
        in_half = self.is_home if side == 'def' else (lambda c: not self.is_home(c))
        table = Duel(self, in_half, key[1], deadline)
        if table.ok:
            self.duel_cost = min(self.duel_cost, time.perf_counter() - t0)
        self.duels[key] = table if table.ok else None
        return self.duels[key]

    def is_home(self, cell):
        return cell[0] < self.mid if self.red else cell[0] >= self.mid

    def find_dead_ends(self):
        """Remove cells with one neighbour until none are left.  The removed
        cells form dead ends; for each we store the exit and how deep it is."""
        degree = {c: len(self.nbrs[c]) for c in self.cells}
        removed = set()
        queue = deque(c for c in self.cells if degree[c] <= 1)
        while queue:
            c = queue.popleft()
            if c in removed:
                continue
            removed.add(c)
            for _, n in self.nbrs[c]:
                if n not in removed:
                    degree[n] -= 1
                    if degree[n] <= 1:
                        queue.append(n)
        self.pocket_exit, self.pocket_depth = {}, {}
        for e in self.cells:
            if e in removed:
                continue
            for _, n in self.nbrs[e]:
                if n in removed and n not in self.pocket_exit:
                    self.pocket_exit[n], self.pocket_depth[n] = e, 1
                    q = deque([(n, 1)])
                    while q:
                        c, d = q.popleft()
                        for _, m in self.nbrs[c]:
                            if m in removed and m not in self.pocket_exit:
                                self.pocket_exit[m], self.pocket_depth[m] = e, d + 1
                                q.append((m, d + 1))

    def bfs(self, start, blocked=None):
        dist = {start: 0}
        q = deque([start])
        while q:
            c = q.popleft()
            for _, n in self.nbrs[c]:
                if n not in dist and not (blocked and blocked(n)):
                    dist[n] = dist[c] + 1
                    q.append(n)
        return dist

    # ----- opponent beliefs

    def update_beliefs(self, agent, game_state):
        """Called by both agents, but only updates once per turn."""
        timeleft = game_state.data.timeleft
        if self.last_update == timeleft:
            return
        moved = (agent.index - 1) % game_state.get_num_agents()   # who just moved
        first_turn = timeleft >= self.initial_time
        self.last_update = timeleft

        team_pos = [game_state.get_agent_position(i) for i in self.agent_ids]
        noisy = game_state.get_agent_distances()
        my_pos = game_state.get_agent_position(agent.index)

        food = set(agent.get_food_you_are_defending(game_state).as_list())
        caps = set(agent.get_capsules_you_are_defending(game_state))
        eaten = (self.prev_food - food) | (self.prev_caps - caps)
        self.prev_food, self.prev_caps = food, caps

        for o in self.opponents:
            st = game_state.get_agent_state(o)
            pos = game_state.get_agent_position(o)
            prev_seen = self.last_seen[o]
            if pos is not None:
                self.beliefs[o] = {pos: 1.0}
                self.last_seen[o] = (pos, st.is_pacman)
                continue
            self.last_seen[o] = None

            # An invader that vanished next to one of our ghosts was eaten.
            if prev_seen is not None and prev_seen[1] and not st.is_pacman:
                p = prev_seen[0]
                for i, tp in zip(self.agent_ids, team_pos):
                    if tp and not game_state.get_agent_state(i).is_pacman \
                            and abs(tp[0] - p[0]) + abs(tp[1] - p[1]) <= 2:
                        self.beliefs[o] = {self.opp_start[o]: 1.0}
                        break

            b = self.beliefs[o]
            if o == moved and not first_turn:
                b = self.spread(b)
            if o == moved and eaten:
                b = {c: 1.0 for c in ([c for c in eaten if c in b] or list(eaten))}
            b = self.observe(b, o, st, my_pos, noisy, team_pos)
            if not b:
                b = self.reset(o, st, my_pos, noisy, team_pos)
            self.beliefs[o] = b

    def spread(self, belief):
        """The opponent moved: share each cell's probability with its neighbours."""
        new_belief = {}
        for cell, prob in belief.items():
            share = prob / (len(self.nbrs[cell]) + 1)
            new_belief[cell] = new_belief.get(cell, 0.0) + share
            for _, nbr in self.nbrs[cell]:
                new_belief[nbr] = new_belief.get(nbr, 0.0) + share
        return new_belief

    def on_right_side(self, cell, opponent):
        """An opponent is a Pacman exactly when it is on our half."""
        return opponent.is_pacman == self.is_home(cell)

    def possible(self, cell, opponent_index, opponent, my_pos, noisy, team_pos):
        """Could this opponent be on this cell, given what we observe?"""
        if not self.on_right_side(cell, opponent):
            return False
        for teammate_pos in team_pos:
            if teammate_pos is not None and \
                    abs(teammate_pos[0] - cell[0]) + abs(teammate_pos[1] - cell[1]) <= SIGHT:
                return False          # we would see it
        if noisy is not None and my_pos is not None:
            true_dist = abs(my_pos[0] - cell[0]) + abs(my_pos[1] - cell[1])
            if abs(noisy[opponent_index] - true_dist) > SONAR_NOISE:
                return False          # does not match the noisy distance
        return True

    def observe(self, belief, opponent_index, opponent, my_pos, noisy, team_pos):
        """Drop impossible cells and renormalise."""
        kept = {cell: prob for cell, prob in belief.items()
                if self.possible(cell, opponent_index, opponent, my_pos, noisy, team_pos)}
        total = sum(kept.values())
        return {cell: prob / total for cell, prob in kept.items()} if total > 0 else {}

    def reset(self, opponent_index, opponent, my_pos, noisy, team_pos):
        """Every cell was ruled out: start again from what is still possible."""
        start = self.opp_start[opponent_index]
        if self.possible(start, opponent_index, opponent, my_pos, noisy, team_pos):
            return {start: 1.0}
        candidates = [cell for cell in self.cells
                      if self.possible(cell, opponent_index, opponent, my_pos, noisy, team_pos)]
        if not candidates:
            candidates = [cell for cell in self.cells
                          if self.on_right_side(cell, opponent)] or self.cells
        return {cell: 1.0 / len(candidates) for cell in candidates}

    def mode(self, o):
        """Most likely cell of opponent o."""
        return max(self.beliefs[o].items(), key=lambda kv: kv[1])[0]


class BaseAgent(CaptureAgent):
    """Everything both agents need: searches, attacking, hunting, recovery."""

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        # The first agent of the team builds the shared data.  If an agent
        # registers twice it is a new game in the same process.
        td = _TEAM_DATA.get(self.red)
        if td is None or self.index in td.registered:
            td = TeamData(self, game_state)
            _TEAM_DATA[self.red] = td
        td.registered.add(self.index)
        self.td = td
        self.target = None
        self.returning = False
        # Recovery: a short detour when we keep circling without progress.
        self.history = deque(maxlen=STALL_WINDOW)   # our recent cells
        self.prev_obs = None                        # (cell, carrying) last turn
        self.stall = 0                              # turns in a row without progress
        self.recovery_target = None
        self.recovery_path = None                   # planned once, then followed
        self.recovery_turns_left = 0
        self.recovery_phase = None                  # 'home', 'enter' or 'cross'
        self.tried_recovery_targets = set()
        self.border_hist = deque(maxlen=6)
        self.standoff = 0
        self.bait_left = 0
        self.help_turns = 0

    def dist(self, a, b):
        return self.get_maze_distance(a, b)

    def my_pos(self, game_state):
        return game_state.get_agent_position(self.index)

    def choose_action(self, game_state):
        """Update beliefs, let the role decide, then let recovery and the
        duel check adjust the move.

        Fallback: if any of this raises an exception or returns an illegal
        move, a legal move is chosen at random, preferring a real move over
        Stop.  (A crash would forfeit the game, but note that this also
        hides bugs: an exception just turns into a random move.)"""
        legal = game_state.get_legal_actions(self.index)
        self.intentional_movement = False   # guarding or chasing on purpose this turn
        self.hunting = False                # chasing an invader this turn
        try:
            self.td.update_beliefs(self, game_state)
            action = self.decide(game_state)
            action = self.recover(game_state, action)
            action = self.duel_filter(game_state, action)
        except Exception:
            action = None
        if action not in legal:
            action = random.choice([a for a in legal if a != Directions.STOP] or legal)
        return action

    def moves_left(self, game_state):
        return (game_state.data.timeleft + 3) // 4

    def home_dist(self, cell):
        return min(self.dist(cell, c) for c in self.td.home_border)

    def step_towards(self, game_state, goals, blocked=None):
        """First move of a shortest path to the nearest goal."""
        my_pos = self.my_pos(game_state)
        goals = set(goals)
        if not goals:
            return None
        if my_pos in goals:
            return Directions.STOP
        first = {my_pos: None}
        q = deque([my_pos])
        while q:
            c = q.popleft()
            for a, n in self.td.nbrs[c]:
                if n in first or (blocked and blocked(n)):
                    continue
                first[n] = a if c == my_pos else first[c]
                if n in goals:
                    return first[n]
                q.append(n)
        return None

    def step_home_side(self, game_state, goal):
        return self.step_towards(game_state, [goal], blocked=lambda c: not self.td.is_home(c))

    def dangerous_ghosts(self, game_state):
        out = []
        for o in self.td.opponents:
            st = game_state.get_agent_state(o)
            p = game_state.get_agent_position(o)
            if p is not None and not st.is_pacman and st.scared_timer <= 1:
                out.append(p)
        return out

    # ----- searches used by the attacker

    def threat_times(self, game_state):
        """For every enemy cell, the earliest move we expect a dangerous ghost
        could be there.  Visible ghosts are measured from where they are;
        unseen ghosts from their most likely cell, plus UNSEEN_SLACK moves,
        so for them this is an estimate.  Also sets self.visible_threat
        (distance to the nearest visible ghost that would still be dangerous
        before we get home)."""
        td = self.td
        my_pos = self.my_pos(game_state)
        times = {}
        self.visible_threat = INF
        trip_home = self.home_dist(my_pos)
        for o in td.opponents:
            st = game_state.get_agent_state(o)
            if st.is_pacman:
                continue
            pos = game_state.get_agent_position(o)
            slack = 0
            if pos is None:
                pos, slack = td.mode(o), UNSEEN_SLACK
            elif st.scared_timer <= trip_home + 2:
                self.visible_threat = min(self.visible_threat, self.dist(my_pos, pos))
            for c, d in td.bfs(pos).items():
                if not td.is_home(c):
                    t = max(d + slack, st.scared_timer)   # harmless while scared
                    if t < times.get(c, INF):
                        times[c] = t
        return times

    def safe_search(self, my_pos, times):
        """BFS over cells we expect to reach strictly before any ghost
        ("safe" in the sense of threat_times: exact for visible ghosts,
        an estimate for unseen ones)."""
        dist, first = {my_pos: 0}, {my_pos: None}
        q = deque([my_pos])
        while q:
            c = q.popleft()
            k = dist[c] + 1
            for a, n in self.td.nbrs[c]:
                if n in dist or times.get(n, INF) <= k:
                    continue
                dist[n] = k
                first[n] = a if c == my_pos else first[c]
                q.append(n)
        return dist, first

    def risky_search(self, my_pos, times):
        """Dijkstra where losing the race to a cell costs extra.  Cells a
        ghost reaches within two moves are still forbidden."""
        cost, first, steps, parent = {my_pos: 0}, {my_pos: None}, {my_pos: 0}, {my_pos: None}
        heap = [(0, 0, my_pos)]
        tie = 0
        while heap:
            c_cost, _, c = heapq.heappop(heap)
            if c_cost > cost[c]:
                continue
            k = steps[c] + 1
            for a, n in self.td.nbrs[c]:
                t = times.get(n, INF)
                margin = t - k
                if margin <= 0 and t <= 2:
                    continue
                nc = c_cost + 1 + (RISK_PENALTY * (2 - margin) if margin < 2 else 0)
                if nc < cost.get(n, INF):
                    cost[n], steps[n], parent[n] = nc, k, c
                    first[n] = a if c == my_pos else first[c]
                    tie += 1
                    heapq.heappush(heap, (nc, tie, n))
        return cost, first, steps, parent

    def pocket_deficit(self, cell, arrive, times):
        """How many moves too late we would get out of a dead end (0 = fine)."""
        exit_ = self.td.pocket_exit.get(cell)
        if exit_ is None:
            return 0
        return max(0, arrive + self.td.pocket_depth[cell] + 2 - times.get(exit_, INF))

    def pocket_safe(self, cell, arrive, times):
        exit_ = self.td.pocket_exit.get(cell)
        if exit_ is None:
            return True
        return times.get(exit_, INF) > arrive + self.td.pocket_depth[cell] + 1

    def no_ghost_threat(self, game_state, margin):
        """No enemy ghost that stays (or becomes) dangerous within `margin` moves."""
        for o in self.td.opponents:
            st = game_state.get_agent_state(o)
            if not st.is_pacman and st.scared_timer <= margin:
                return False
        return True

    # ----- attacking

    def attack(self, game_state, force_home=False, avoid=None):
        """The attacker's decision, in five steps (checked in this order):
            1. catch a nearby invader      (only while still at home)
            2. decide whether to bank      (and if so, head home)
            3. look for safe food
            4. consider a risky route      (only when carrying little)
            5. try another crossing        (nothing reachable: we are blocked)"""
        td = self.td
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.my_pos(game_state)
        carrying = my_state.num_carrying
        food = self.get_food(game_state).as_list()
        caps = self.get_capsules(game_state)
        if avoid is not None and len(food) > 4:
            # We are the second attacker: stay away from the teammate.
            food = [f for f in food if self.dist(f, avoid) > 4] or food

        # --- 1. Catch a nearby invader
        at_home_and_able = not my_state.is_pacman and my_state.scared_timer == 0
        if at_home_and_able:
            nearby = [(self.dist(my_pos, p), p) for _, p in self.visible_invaders(game_state)]
            nearby = [x for x in nearby if x[0] <= 4]
            if nearby:
                self.intentional_movement = self.hunting = True
                a = self.hunt(game_state, min(nearby)[1], False)
                if a is not None:
                    return a

        times = self.threat_times(game_state)
        dist, first = self.safe_search(my_pos, times)
        home_cells = [c for c in td.home_border if c in dist]
        home_d = min((dist[c] for c in home_cells), default=INF)
        ghost_watching = self.visible_threat <= 5

        # --- 2. Decide whether to bank
        go_home = (carrying > 0 and self.should_bank(game_state, food)) or force_home
        self.returning = go_home and carrying > 0 and my_state.is_pacman
        if go_home:
            goals = home_cells
            if ghost_watching:              # a closer safe capsule is as good
                goals = goals + [c for c in caps if c in dist and dist[c] < home_d]
            if goals:
                g = min(goals, key=lambda c: dist[c])
                return first[g] if g != my_pos else Directions.STOP
            return self.risky_home(game_state, times, caps)

        # --- 3. Look for safe food (or a capsule when a ghost is close)
        candidates = [(dist[f], f) for f in food
                      if f in dist and self.pocket_safe(f, dist[f], times)]
        if ghost_watching:
            candidates += [(dist[c] - 3, c) for c in caps if c in dist]
        if candidates:
            best_d, best = min(candidates)
            dist_of = {f: d for d, f in candidates}
            if self.target in dist_of and dist_of[self.target] <= best_d + 2:
                best = self.target          # do not switch targets for a small gain
            self.target = best
            return first[best] if best != my_pos else Directions.STOP

        # --- 4. Consider a risky route (little to lose)
        little_to_lose = carrying <= RISK_CARRY
        if little_to_lose:
            cost, risky_first, steps, _ = self.risky_search(my_pos, times)
            options = [(cost[f] + POCKET_PENALTY * self.pocket_deficit(f, steps[f], times), f)
                       for f in food if f in cost]
            if options:
                best = min(options)[1]
                self.target = best
                return risky_first[best] if best != my_pos else Directions.STOP

        # --- 5. Nothing to eat: go home if we are out, else try another crossing
        self.target = None
        if my_state.is_pacman or carrying > 0:
            if home_cells:
                g = min(home_cells, key=lambda c: dist[c])
                return first[g] if g != my_pos else Directions.STOP
            if my_state.is_pacman:
                return self.risky_home(game_state, times, caps)
        self.blocked = True                 # tells the defender we cannot get in
        best = max(td.crossings, key=lambda c: self.cross_margin(c, times) - self.dist(my_pos, c))
        return self.step_home_side(game_state, best)

    def should_bank(self, game_state, food):
        """Should we take the food we carry home now?  Checked in order:
          - the game is nearly over (food or time);
          - we carry CARRY_LIMIT and some enemy ghost could still defend;
          - a dangerous ghost is watching us (unless it is the only enemy
            ghost and the duel table shows we can always escape it);
          - we are already returning and the route home is still threatened.
        So the return is not unconditional: once the threat is gone, the
        attacker can go back to collecting."""
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        trip_home = self.home_dist(self.my_pos(game_state))
        game_ending = len(food) <= 2 or self.moves_left(game_state) <= trip_home + 4
        if game_ending:
            return True
        full_and_defended = carrying >= CARRY_LIMIT and not self.no_ghost_threat(game_state, trip_home + 2)
        if full_and_defended:
            return True
        if self.visible_threat <= 5 and not (my_state.is_pacman and self.duel_won(game_state)):
            return True
        still_threatened = self.returning and my_state.is_pacman \
            and not self.no_ghost_threat(game_state, trip_home + 2)
        return still_threatened

    def cross_margin(self, crossing, times):
        x, y = crossing
        return min(times.get((x + 1, y) if self.red else (x - 1, y), INF), 50)

    def risky_home(self, game_state, times, caps):
        my_pos = self.my_pos(game_state)
        cost, first, _, _ = self.risky_search(my_pos, times)
        goals = [c for c in list(self.td.home_border) + list(caps) if c in cost and c != my_pos]
        if goals:
            return first[min(goals, key=lambda c: cost[c])]
        # Cornered: move to where a ghost needs longest to reach us.
        best, best_score = None, None
        for a, n in self.td.nbrs[my_pos]:
            score = (times.get(n, INF), -self.home_dist(n))
            if best_score is None or score > best_score:
                best, best_score = a, score
        return best

    def duel_won(self, game_state):
        """True if only one enemy ghost is active and we can force an escape from it."""
        td = self.td
        table = td.duel('att', self.get_capsules(game_state))
        if table is None:
            return False
        active = [o for o in td.opponents if not game_state.get_agent_state(o).is_pacman
                  and game_state.get_agent_state(o).scared_timer <= 1]
        if len(active) != 1:
            return False             # the table is only exact for one ghost
        pos = self.my_pos(game_state)
        for g in self.dangerous_ghosts(game_state):
            got = table.get(g, pos, True)
            if got is not None and got[0] != Duel.PAC:
                return False
        return True

    def duel_filter(self, game_state, action):
        """As a Pacman next to a visible ghost, do not walk into a forced
        capture if another move avoids it.  Each move gets a class:
        3 home, 2 forced escape, 1 cannot be caught soon, 0 caught soon,
        -1 onto the ghost."""
        td = self.td
        if not game_state.get_agent_state(self.index).is_pacman or self.stall >= DUEL_PATIENCE:
            return action            # (after circling for long, take the risk)
        ghosts = self.dangerous_ghosts(game_state)
        table = td.duel('att', self.get_capsules(game_state)) if ghosts else None
        if table is None:
            return action
        pos = self.my_pos(game_state)
        scores = {}
        for a, n in td.nbrs[pos] + [(Directions.STOP, pos)]:
            worst = (3, 0)
            if not td.is_home(n):
                for g in ghosts:
                    got = table.get(g, n, False)
                    if got is None:
                        continue     # that ghost is not on its own half
                    r, d = got
                    if n == g:
                        cls = (-1, 0)
                    elif r == Duel.PAC:
                        cls = (2, -d)
                    elif r == Duel.DRAW or d > DUEL_DEPTH:
                        cls = (1, d if r == Duel.GHOST else INF)
                    else:
                        cls = (0, d)
                    worst = min(worst, cls)
            scores[a] = worst
        mine = scores.get(action)
        if mine is None or mine[0] >= 1 or mine == max(scores.values()):
            return action
        return max(scores, key=lambda a: scores[a])

    # ----- defending

    def visible_invaders(self, game_state):
        out = []
        for o in self.td.opponents:
            p = game_state.get_agent_position(o)
            if game_state.get_agent_state(o).is_pacman and p is not None:
                out.append((o, p))
        return out

    def invaders(self, game_state):
        """(opponent, position, carrying) of every invader; unseen ones at
        their most likely cell."""
        out = []
        for o in self.td.opponents:
            st = game_state.get_agent_state(o)
            if st.is_pacman:
                p = game_state.get_agent_position(o)
                out.append((o, p if p is not None else self.td.mode(o), st.num_carrying))
        return out

    def defend(self, game_state, exclude=None):
        """Chase the closest invader (preferably not the teammate's)."""
        invs = self.invaders(game_state)
        if not invs:
            return None
        my_pos = self.my_pos(game_state)
        pool = [i for i in invs if i[0] != exclude] or invs
        o, target, _ = min(pool, key=lambda i: self.dist(my_pos, i[1]))
        self.td.defend_target[self.index] = o
        self.intentional_movement = self.hunting = True
        return self.hunt(game_state, target, game_state.get_agent_state(self.index).scared_timer > 0)

    def mate_target(self):
        mate = [i for i in self.td.agent_ids if i != self.index]
        return self.td.defend_target.get(mate[0]) if mate else None

    def hunt(self, game_state, target, scared):
        """Go for an invader.  While scared, stay two cells away instead."""
        td = self.td
        my_pos = self.my_pos(game_state)
        if not scared:
            return self.duel_hunt(game_state, target) or self.step_home_side(game_state, target)
        best, best_score = None, None
        for a, n in td.nbrs[my_pos] + [(Directions.STOP, my_pos)]:
            if td.is_home(n):
                d = self.dist(n, target)
                score = (d <= 1, abs(d - 2))
                if best_score is None or score < best_score:
                    best, best_score = a, score
        return best

    def duel_hunt(self, game_state, target):
        """Best move against a visible invader using the solved duel: catch it
        as fast as possible, else keep it from getting home, else delay it."""
        td = self.td
        if target not in [p for _, p in self.visible_invaders(game_state)]:
            return None
        table = td.duel('def', self.get_capsules_you_are_defending(game_state))
        my_pos = self.my_pos(game_state)
        if table is None or table.get(my_pos, target, False) is None:
            return None
        best, best_key = None, None
        for a, n in td.nbrs[my_pos] + [(Directions.STOP, my_pos)]:
            if not td.is_home(n):
                continue
            if n == target:
                return a
            r, d = table.get(n, target, True)
            if r == Duel.GHOST:
                key = (0, d, 0)
            elif r == Duel.DRAW:
                key = (1, self.dist(n, target), a == Directions.STOP)
            else:
                key = (2, -d, self.dist(n, target))
            if best_key is None or key < best_key:
                best, best_key = a, key
        return best

    # ----- getting unstuck

    def recover(self, game_state, action):
        """If we keep revisiting the same cells without eating, banking or
        dying (and are not guarding or chasing on purpose), take a detour:
        commit to a new target and the path to it for a few turns.

        A detour starts after STALL_TURNS such turns.  It ends when we make
        progress, reach the target, or COMMIT_TURNS run out; reaching a
        'cross' target turns it into an 'enter' detour from there."""
        my_state = game_state.get_agent_state(self.index)
        pos = self.my_pos(game_state)
        progressed = False
        if self.prev_obs is not None:
            ppos, pcarry = self.prev_obs
            died = pos == self.start and abs(ppos[0] - pos[0]) + abs(ppos[1] - pos[1]) > 1
            progressed = died or my_state.num_carrying != pcarry
        self.prev_obs = (pos, my_state.num_carrying)

        if progressed or self.intentional_movement:
            self.end_recovery()
            self.stall = 0
            self.history.clear()
        elif pos in self.history:
            self.stall += 1
        else:
            self.stall = 0
            self.tried_recovery_targets = set()
        self.history.append(pos)

        if self.recovery_target is not None:
            if pos == self.recovery_target and self.recovery_phase == 'cross':
                # Reached the new crossing: now try to get in from here.
                self.tried_recovery_targets.add(self.recovery_target)
                self.recovery_target = None
                if not self.choose_entry(game_state, pos, self.danger_cells(game_state)):
                    self.end_recovery()
                    return action
            elif pos == self.recovery_target or self.recovery_turns_left <= 0:
                self.end_recovery()
                self.stall = 0
                self.history.clear()
                self.history.append(pos)
                return action
        elif self.stall < STALL_TURNS:
            return action

        step = self.recovery_step(game_state, pos)
        if step is None:
            self.end_recovery()
            return action
        self.recovery_turns_left -= 1
        return step

    def end_recovery(self):
        if self.recovery_target is not None:
            self.tried_recovery_targets.add(self.recovery_target)
        self.recovery_target, self.recovery_turns_left, self.recovery_phase, self.recovery_path = None, 0, None, None

    def danger_cells(self, game_state):
        """Cells next to a visible enemy that could eat us there."""
        td = self.td
        scared = game_state.get_agent_state(self.index).scared_timer > 0
        bad = set()
        for o in td.opponents:
            st = game_state.get_agent_state(o)
            p = game_state.get_agent_position(o)
            if p is None:
                continue
            if not st.is_pacman and st.scared_timer <= 1:
                on_home = False      # a ghost is dangerous on its side
            elif st.is_pacman and scared:
                on_home = True       # an invader is dangerous to us while scared
            else:
                continue
            for c in [p] + [n for _, n in td.nbrs[p]]:
                if td.is_home(c) == on_home:
                    bad.add(c)
        return bad

    def recovery_step(self, game_state, pos):
        bad = self.danger_cells(game_state)
        for _ in range(4):
            if self.recovery_target is None and not self.choose_recovery_target(game_state, pos, bad):
                return None
            step = self.follow_path(game_state, pos, bad)
            if step is not None:
                return step
            self.tried_recovery_targets.add(self.recovery_target)   # blocked or unsafe: try another
            self.recovery_target = None
        return None

    def follow_path(self, game_state, pos, bad):
        """Next step along the path to the recovery target.  The path is
        planned once and then followed, because re-planning every turn
        against a ghost that copies our moves just recreates the loop."""
        td = self.td
        target = self.recovery_target
        path = self.recovery_path
        if not (path and path[0] == pos and len(path) > 1 and path[1] not in bad):
            if game_state.get_agent_state(self.index).is_pacman or self.recovery_phase == 'enter':
                _, _, _, parent = self.risky_search(pos, self.threat_times(game_state))
                if target not in parent:
                    return None
                path, c = [], target
                while c is not None:
                    path.append(c)
                    c = parent[c]
                path.reverse()
            else:
                dist = td.bfs(target, blocked=lambda c: c in bad or (not td.is_home(c) and c != target))
                if pos not in dist:
                    return None
                path, c = [pos], pos
                while c != target:
                    c = min((n for _, n in td.nbrs[c] if n in dist), key=lambda n: dist[n])
                    path.append(c)
            if len(path) < 2 or path[1] in bad:
                return None
        self.recovery_path = path[1:]
        for a, n in td.nbrs[pos]:
            if n == path[1]:
                return a
        return None

    def choose_recovery_target(self, game_state, pos, bad):
        """Pacman with food: go home by a crossing outside the loop.
        Otherwise: go for food; if there is none, try a crossing further away."""
        td = self.td
        my_state = game_state.get_agent_state(self.index)
        self.recovery_path = None
        recent = set(self.history)
        if my_state.is_pacman:
            # Carrying nothing, going home would only repeat the loop.
            if my_state.num_carrying == 0 and self.choose_entry(game_state, pos, bad):
                return True
            cost, _, _, _ = self.risky_search(pos, self.threat_times(game_state))
            goals = [c for c in list(td.home_border) + list(self.get_capsules(game_state))
                     if c in cost and c not in self.tried_recovery_targets and c not in bad and c != pos]
            goals = [c for c in goals if c not in recent] or goals
            if not goals:
                return False
            self.recovery_target, self.recovery_phase = min(goals, key=lambda c: cost[c]), 'home'
        else:
            if self.choose_entry(game_state, pos, bad):
                return True
            times = self.threat_times(game_state)
            home_d = td.bfs(pos, blocked=lambda c: not td.is_home(c))
            opts = []
            for c in td.crossings:
                if c in home_d and home_d[c] >= 3 and c not in recent \
                        and c not in self.tried_recovery_targets and c not in bad:
                    opts.append((self.cross_margin(c, times) - home_d[c] - 1, -home_d[c], c))
            if not opts:
                return False
            self.recovery_target, self.recovery_phase = max(opts)[2], 'cross'
        self.recovery_turns_left = COMMIT_TURNS
        return True

    def choose_entry(self, game_state, pos, bad):
        """Commit to the food with the cheapest risk-weighted route."""
        self.recovery_path = None
        times = self.threat_times(game_state)
        cost, first, steps, _ = self.risky_search(pos, times)
        opts = [(cost[f] + POCKET_PENALTY * self.pocket_deficit(f, steps[f], times), f)
                for f in self.get_food(game_state).as_list()
                if f in cost and f not in self.tried_recovery_targets and f not in bad
                and first.get(f) not in (None, Directions.STOP)]
        if not opts:
            return False
        self.recovery_target, self.recovery_phase, self.recovery_turns_left = min(opts)[1], 'enter', COMMIT_TURNS
        return True


class AttackAgent(BaseAgent):
    """Collects food; comes back to help when the invasion is serious."""

    def decide(self, game_state):
        self.blocked = False
        self.td.defend_target[self.index] = None
        action = self.help_defend(game_state)
        if action is None:
            action = self.attack(game_state)

        # Tell the defender how long we have been stuck at the border.
        st = game_state.get_agent_state(self.index)
        pos = self.my_pos(game_state)
        revisit = pos in self.border_hist
        self.border_hist.append(pos)
        if st.is_pacman or st.num_carrying or self.hunting:
            self.td.blocked_turns = 0
        elif self.blocked or (revisit and min(self.dist(pos, c) for c in self.td.crossings) <= 2):
            self.td.blocked_turns += 1
        self.td.attack_target = self.target
        return action

    def help_defend(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        invs = self.invaders(game_state)
        if not invs or sum(i[2] for i in invs) < HELP_CARRY or my_state.num_carrying > 2:
            self.help_turns = 0
            return None
        if my_state.is_pacman:
            return self.attack(game_state, force_home=True)
        if my_state.scared_timer > 0:
            return None
        self.help_turns += 1
        if self.help_turns > HELP_MAX_TURNS:
            return None              # do not guard one invader forever
        return self.defend(game_state, exclude=self.mate_target())


class DefendAgent(BaseAgent):
    """Chases invaders, otherwise guards the border.  Joins the attack when
    we are not ahead and our attacker is stuck at the border."""

    def decide(self, game_state):
        td = self.td
        my_state = game_state.get_agent_state(self.index)
        my_pos = self.my_pos(game_state)
        invaders = self.invaders(game_state)

        if invaders or self.get_score(game_state) > 0:
            td.doubling = False
        elif td.blocked_turns >= DOUBLE_AFTER:
            td.doubling = True
        if td.doubling:
            return self.attack(game_state, avoid=td.attack_target)

        if my_state.is_pacman:
            return self.attack(game_state, force_home=True)

        if invaders:
            self.standoff = self.bait_left = 0
            a = self.defend(game_state, exclude=self.mate_target())
            if a is not None:
                return a

        # Guard the border cell closest to the opponent closest to crossing.
        threat = min(td.opponents, key=lambda o: min(self.dist(td.mode(o), c) for c in td.crossings))
        mode = td.mode(threat)
        guard = min(td.home_border, key=lambda c: (self.dist(mode, c), self.dist(my_pos, c)))
        self.intentional_movement = True

        # If a ghost waits right in front of us, back off for a while so it
        # comes in (otherwise both sides just stand there).
        if self.bait_left > 0:
            self.bait_left -= 1
            bait = self.bait_cell(guard)
            if bait is not None and my_pos != bait:
                a = self.step_home_side(game_state, bait)
                if a is not None:
                    return a
            return Directions.STOP
        facing = [p for p in self.enemy_ghosts(game_state) if self.dist(my_pos, p) <= 2]
        self.standoff = self.standoff + 1 if (facing and my_pos == guard) else 0
        if self.standoff >= GUARD_PATIENCE:
            self.standoff = 0
            self.bait_left = BAIT_TURNS
            bait = self.bait_cell(guard)
            if bait is not None:
                a = self.step_home_side(game_state, bait)
                if a is not None:
                    return a

        if my_pos == guard:
            return Directions.STOP
        return self.step_home_side(game_state, guard)

    def enemy_ghosts(self, game_state):
        out = []
        for o in self.td.opponents:
            p = game_state.get_agent_position(o)
            if p is not None and not game_state.get_agent_state(o).is_pacman:
                out.append(p)
        return out

    def bait_cell(self, guard):
        """A home cell BAIT_DIST moves behind the guard cell."""
        td = self.td
        d = td.bfs(guard, blocked=lambda c: not td.is_home(c))
        far = [c for c, k in d.items() if k >= BAIT_DIST]
        if not far:
            return None
        return min(far, key=lambda c: (d[c], abs(c[0] - td.mid)))
