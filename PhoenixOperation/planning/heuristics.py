from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects, get_all_groundings


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###
    unsatisfied = set(goal - state)

    if not unsatisfied:
        return 0

    ground_actions = get_all_groundings(domain, objects)
    cost = 0

    while unsatisfied:
        best_covered = set()

        for action in ground_actions:
            covered = set(action.add_list) & unsatisfied

            if len(covered) > len(best_covered):
                best_covered = covered

        if not best_covered:
            return float("inf")

        unsatisfied -= best_covered
        cost += 1

    return cost
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###
    # -----------------------------------------------------------------------
    # 
    #
    # Primera versión:
    #
    # current_state = set(state)
    # steps = 0
    #
    # while not goal.issubset(current_state):
    #     for action in _all_ground_actions(domain, objects):
    #         current_state = current_state | set(action["add_list"])
    #         steps += 1
    #
    # return steps
    #
    # Problema detectado en la primera versión:
    # Esta primera idea estaba incompleta porque aplicaba acciones sin revisar
    # si sus precondiciones se cumplían. Eso se parece más a ignorar
    # precondiciones, pero esta heurística no debe hacer eso. En ignore-delete,
    # las precondiciones siguen importando; lo único que se ignora es la lista
    # de borrado. Además, la versión inicial aplicaba acciones sin escoger la
    # mejor y podía contar pasos innecesarios.
    #
    # Prompt utilizado:
    # "Tengo esta primera versión de ignoreDeleteListsHeuristic, pero creo que
    # estoy confundiendo ignorar delete lists con ignorar precondiciones,no se
    # bien cómo hacer para que las acciones sigan necesitando precondiciones,
    # pero que al aplicarlas no se borre nada me ayudas a corregirla?"
    #
    # -----------------------------------------------------------------------

    current_state = set(state)
    goal_set = set(goal)
    steps = 0

    if goal_set.issubset(current_state):
        return 0

    ground_actions = get_all_groundings(domain, objects)

    while not goal_set.issubset(current_state):
        unsatisfied = goal_set - current_state
        best_action = None
        best_progress = set()

        for action in ground_actions:
            precond_pos = set(action.precond_pos)
            precond_neg = set(action.precond_neg)

            positive_ok = precond_pos.issubset(current_state)
            negative_ok = precond_neg.isdisjoint(current_state)

            if not positive_ok or not negative_ok:
                continue

            progress = set(action.add_list) & unsatisfied

            if len(progress) > len(best_progress):
                best_action = action
                best_progress = progress

        if best_action is None or not best_progress:
            return float("inf")

        current_state |= set(best_action.add_list)
        steps += 1

    return steps
    ### End of your code ###