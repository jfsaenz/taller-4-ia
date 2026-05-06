from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects, get_all_groundings


_GROUNDINGS_CACHE = {}


def _cached_groundings(
    domain: list[ActionSchema],
    objects: Objects,
):
    """
    Guarda las acciones aterrizadas para no volver a generarlas cada vez que
    se llama una heurística.
    """
    key = (id(domain), id(objects))

    if key not in _GROUNDINGS_CACHE:
        _GROUNDINGS_CACHE[key] = get_all_groundings(domain, objects)

    return _GROUNDINGS_CACHE[key]


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

    ground_actions = _cached_groundings(domain, objects)
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

    In this monotone relaxation, the state only grows over time.
    """
    ### Your code here ###
    # -----------------------------------------------------------------------
    # Registro de uso de IA - ignoreDeleteListsHeuristic
    #
    # Primera versión tentativa:
    #
    # current_state = set(state)
    # steps = 0
    #
    # while not goal.issubset(current_state):
    #     for action in get_all_groundings(domain, objects):
    #         if action.precond_pos.issubset(current_state):
    #             current_state = current_state | set(action.add_list)
    #             steps += 1
    #
    # return steps
    #
    # Problema que encontré:
    # Esta idea funcionaba en mapas pequeños, pero en mapas medianos se quedaba
    # muy lenta. El problema era que A* llama muchas veces la heurística y en
    # cada llamada se volvían a revisar muchas acciones. Además, contar una
    # acción por vez hacía que el cálculo avanzara muy despacio.
    #
    # Prompt utilizado:
    # "La heurística ignoreDeleteLists me funciona en tinyBase, pero en
    # mediumRescue y warehouseRescue se queda pegada, creo que estoy haciendo
    # el cálculo muy pesado, no entiendo cómo hacerla más rápida sin cambiar
    # la idea de ignorar las delete lists, me ayudas a revisarla"
    #
    # Después de revisar:
    # Dejé una versión por niveles. En cada nivel se revisan las acciones que
    # sí cumplen sus precondiciones, se agregan sus efectos positivos y no se
    # borra ningún fluente. También se usa caché para no generar otra vez todas
    # las acciones aterrizadas en cada llamada.
    # -----------------------------------------------------------------------

    current_state = set(state)
    goal_set = set(goal)

    if goal_set.issubset(current_state):
        return 0

    ground_actions = _cached_groundings(domain, objects)
    levels = 0

    while not goal_set.issubset(current_state):
        new_fluents = set()

        for action in ground_actions:
            positive_ok = action.precond_pos.issubset(current_state)
            negative_ok = action.precond_neg.isdisjoint(current_state)

            if positive_ok and negative_ok:
                new_fluents |= set(action.add_list)

        before_size = len(current_state)
        current_state |= new_fluents
        levels += 1

        if len(current_state) == before_size:
            return float("inf")

        if levels > 100:
            return float("inf")

    return levels
    ### End of your code ###