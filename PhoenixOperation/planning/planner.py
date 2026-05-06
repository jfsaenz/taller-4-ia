from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    # -----------------------------------------------------------------------
    #
    # Primera versión 
    #
    # state = problem.getStartState()
    #
    # while not problem.isGoalState(state):
    #     successors = problem.getSuccessors(state)
    #     state = successors[0][0]
    #
    # return []
    #
    # Problema detectado en la primera versión:
    # Esta primera idea no era realmente BFS. Solo tomaba el primer sucesor y
    # avanzaba por ahí, como si fuera una búsqueda greedy sin memoria. Además,
    # no guardaba el plan de acciones, no revisaba estados visitados y podía
    # quedarse dando vueltas si volvía al mismo estado.
    #
    # Prompt utilizado:
    # "Tengo esta primera versión de forwardBFS, pero creo que no estoy haciendo
    # búsqueda en anchura de verdad no se cómo guardar el plan mientras voy
    # explorando estados ni cómo evitar repetir estados ayudame a
    # corregirla usando una cola"
    # -----------------------------------------------------------------------

    start_state = problem.getStartState()

    if problem.isGoalState(start_state):
        return []

    frontier = Queue()
    frontier.push((start_state, []))

    visited = {start_state}

    while not frontier.isEmpty():
        current_state, current_plan = frontier.pop()

        for next_state, action, _cost in problem.getSuccessors(current_state):
            if next_state in visited:
                continue

            new_plan = current_plan + [action]

            if problem.isGoalState(next_state):
                return new_plan

            visited.add(next_state)
            frontier.push((next_state, new_plan))

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    # -----------------------------------------------------------------------
    # Primera versión 
    #
    # return goal_set | action.precond_pos
    #
    # Problema detectado en la primera versión:
    # La primera idea simplemente agregaba las precondiciones de la acción, pero
    # no revisaba si la acción realmente servía para alcanzar algo del objetivo.
    # Tampoco quitaba del objetivo los fluentes que la acción ya logra con su
    # add_list. Además, no revisaba si la acción borraba algo que todavía se
    # necesita como meta.
    #
    # Prompt utilizado:
    # "Estoy intentando hacer regress, pero no entiendo bien por qué no basta
    # con sumar las precondiciones de la acción al objetivo me puedes ayudar
    # a revisar cuándo una acción sirve para regresar un objetivo y cómo se
    # calcula el nuevo objetivo
    # -----------------------------------------------------------------------

    # La acción debe aportar algo al objetivo actual.
    if not action.add_list & goal_set:
        return None

    # La acción no puede borrar algo que todavía necesito que sea verdadero.
    if action.del_list & goal_set:
        return None

    regressed_goal = (goal_set - action.add_list) | action.precond_pos

    # Si una precondición negativa contradice algo que necesito verdadero,
    # esta acción no sirve para este objetivo regresado.
    if action.precond_neg & regressed_goal:
        return None

    return frozenset(regressed_goal)
    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    ### Your code here ###
    # -----------------------------------------------------------------------
    # Primera versión
    #
    # current_goal = problem.goal
    # actions = get_all_groundings(problem.domain, problem.objects)
    #
    # for action in actions:
    #     new_goal = regress(current_goal, action)
    #     if new_goal is not None:
    #         return [action]
    #
    # return []
    #
    # Problema detectado en la primera versión:
    # Esta primera versión solo probaba una regresión y retornaba una acción,
    # pero backward planning necesita seguir regresando varios pasos hasta que
    # el objetivo parcial sea satisfecho por el estado inicial. También faltaba
    # guardar el plan en el orden correcto y evitar visitar los mismos objetivos
    # parciales repetidamente.
    #
    # Prompt utilizado:
    # "Tengo esta primera versión de backwardSearch, pero se queda buscando
    # mucho y no entiendo cómo limitar la regresión. Creo que está probando
    # demasiadas acciones que no sirven. ¿Me puedes ayudar a filtrar mejor las
    # acciones relevantes y evitar objetivos parciales demasiado grandes?"
    # -----------------------------------------------------------------------

    initial_state = problem.initial_state
    start_goal = problem.goal

    if start_goal.issubset(initial_state):
        return []

    all_actions = get_all_groundings(problem.domain, problem.objects)
    static_predicates = {"MedicalPost", "Adjacent", "Pickable"}

    def has_false_static_fluent(goal_description: State) -> bool:
        for fluent in goal_description:
            if len(fluent) == 0:
                continue

            if fluent[0] in static_predicates and fluent not in initial_state:
                return True

        return False

    frontier = PriorityQueue()
    frontier.push((start_goal, []), len(start_goal))

    visited = {start_goal}
    max_plan_depth = 40

    while not frontier.isEmpty():
        current_goal, current_plan = frontier.pop()

        if current_goal.issubset(initial_state):
            return current_plan

        if len(current_plan) >= max_plan_depth:
            continue

        unsatisfied_goal = current_goal - initial_state

        # Acciones relevantes: deben lograr algo que todavía falta.
        relevant_actions = [
            action
            for action in all_actions
            if action.add_list & unsatisfied_goal
        ]

        # Probar primero las acciones que cubren más fluentes pendientes.
        relevant_actions.sort(
            key=lambda action: len(action.add_list & unsatisfied_goal),
            reverse=True,
        )

        for action in relevant_actions:
            new_goal = regress(current_goal, action)

            if new_goal is None:
                continue

            if has_false_static_fluent(new_goal):
                continue

            if new_goal in visited:
                continue

            new_plan = [action] + current_plan
            visited.add(new_goal)

            # Prioridad: preferir objetivos parciales pequeños y planes cortos.
            priority = len(new_goal - initial_state) + len(new_plan)
            frontier.push((new_goal, new_plan), priority)

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###
    # -----------------------------------------------------------------------
    # Registro de uso de IA - aStarPlanner
    #
    # Primera versión tentativa:
    #
    # frontier = PriorityQueue()
    # frontier.push(problem.getStartState(), 0)
    #
    # while not frontier.isEmpty():
    #     state = frontier.pop()
    #     if problem.isGoalState(state):
    #         return []
    #
    # return []
    #
    # Problema que encontré:
    # Esta versión solo metía estados en la cola de prioridad, pero no guardaba
    # el plan que llevaba hasta cada estado. Tampoco guardaba el costo acumulado
    # y si encontraba el objetivo devolvía una lista vacía, aunque sí hubiera un
    # plan. Además, la heurística se podía recalcular muchas veces para estados
    # repetidos.
    #
    # Prompt utilizado:
    # "Tengo A* funcionando en tinyBase, pero cuando uso heurísticas en mapas
    # medianos se demora mucho, no sé si estoy recalculando demasiado la
    # heurística, tampoco estoy seguro de cómo guardar el estado, el plan y el
    # costo al mismo tiempo, me ayudas a revisarlo"
    #
    # Después de revisar:
    # Cada elemento de la frontera guarda el estado actual, el plan y el costo
    # acumulado. La prioridad se calcula con g + h. También se guarda el mejor
    # costo encontrado para cada estado y se usa h_cache para no recalcular la
    # heurística si el mismo estado vuelve a aparecer.
    # -----------------------------------------------------------------------

    start_state = problem.getStartState()

    if problem.isGoalState(start_state):
        return []

    frontier = PriorityQueue()
    best_g_cost: dict[State, int] = {start_state: 0}
    h_cache: dict[State, float] = {}

    def h(state: State) -> float:
        if state not in h_cache:
            h_cache[state] = heuristic(
                state,
                problem.goal,
                problem.domain,
                problem.objects,
            )
        return h_cache[state]

    frontier.push((start_state, [], 0), h(start_state))

    while not frontier.isEmpty():
        current_state, current_plan, current_cost = frontier.pop()

        if current_cost > best_g_cost.get(current_state, float("inf")):
            continue

        if problem.isGoalState(current_state):
            return current_plan

        for next_state, action, step_cost in problem.getSuccessors(current_state):
            new_cost = current_cost + step_cost

            if new_cost >= best_g_cost.get(next_state, float("inf")):
                continue

            best_g_cost[next_state] = new_cost
            new_plan = current_plan + [action]
            priority = new_cost + h(next_state)

            frontier.push((next_state, new_plan, new_cost), priority)

    return []
    ### End of your code ###

# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner