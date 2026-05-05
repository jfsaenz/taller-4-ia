from __future__ import annotations

from collections import deque

from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------


def _initial_state(problem: Problem):
    """Obtiene el estado inicial usando nombres comunes dentro del proyecto."""
    if hasattr(problem, "initial_state"):
        return problem.initial_state
    if hasattr(problem, "initial"):
        return problem.initial
    if hasattr(problem, "start_state"):
        return problem.start_state
    raise AttributeError("No se encontró el estado inicial del problema.")


def _goal_state(problem: Problem):
    """Obtiene el objetivo usando nombres comunes dentro del proyecto."""
    if hasattr(problem, "goal"):
        return problem.goal
    if hasattr(problem, "goal_state"):
        return problem.goal_state
    return None


def _is_goal(problem: Problem, state) -> bool:
    """Verifica si un estado satisface el objetivo del problema."""
    if hasattr(problem, "is_goal"):
        return problem.is_goal(state)
    if hasattr(problem, "goal_test"):
        return problem.goal_test(state)
    if hasattr(problem, "is_goal_state"):
        return problem.is_goal_state(state)

    goal = _goal_state(problem)
    if goal is None:
        return False

    return set(goal).issubset(set(state))


def _make_action(
    name: str,
    args: tuple,
    precond_pos: list[tuple],
    precond_neg: list[tuple],
    add_list: list[tuple],
    del_list: list[tuple],
) -> Action:
    """
    Crea una acción primitiva aterrizada.

    Se dejan varios intentos para adaptarse a la firma de Action usada
    en planning/pddl.py.
    """
    try:
        return Action(
            name=name,
            args=args,
            precond_pos=frozenset(precond_pos),
            precond_neg=frozenset(precond_neg),
            add_list=frozenset(add_list),
            del_list=frozenset(del_list),
        )
    except TypeError:
        pass

    try:
        return Action(
            name,
            args,
            frozenset(precond_pos),
            frozenset(precond_neg),
            frozenset(add_list),
            frozenset(del_list),
        )
    except TypeError:
        pass

    try:
        return Action(
            name=name,
            parameters=list(args),
            precond_pos=frozenset(precond_pos),
            precond_neg=frozenset(precond_neg),
            add_list=frozenset(add_list),
            del_list=frozenset(del_list),
        )
    except TypeError:
        pass

    action = Action.__new__(Action)
    action.name = name
    action.args = args
    action.parameters = list(args)
    action.precond_pos = frozenset(precond_pos)
    action.precond_neg = frozenset(precond_neg)
    action.add_list = frozenset(add_list)
    action.del_list = frozenset(del_list)
    return action


def _simulate_plan(problem: Problem, plan: list[Action]) -> bool:
    """
    Ejecuta un plan desde el estado inicial y verifica si llega al objetivo.
    Si alguna acción no es aplicable, el plan no sirve.
    """
    current_state = _initial_state(problem)

    for action in plan:
        if not is_applicable(current_state, action):
            return False
        current_state = apply_action(current_state, action)

    return _is_goal(problem, current_state)


def _find_first_hla_index(plan: list[Action | HLA]) -> int | None:
    """Retorna la posición de la primera HLA dentro del plan."""
    for i, step in enumerate(plan):
        if not is_primitive(step):
            return i
    return None


def _plan_key(plan: list[Action | HLA]) -> tuple:
    """Convierte un plan en una llave comparable para evitar repetir planes."""
    return tuple(repr(step) for step in plan)


def _extract_fluents(state, name: str) -> list[tuple]:
    """Extrae fluentes de un estado según el nombre del predicado."""
    return [fluent for fluent in state if len(fluent) > 0 and fluent[0] == name]


def _find_location(state, obj):
    """Busca la ubicación de un objeto a partir de fluentes At(obj, loc)."""
    for fluent in state:
        if len(fluent) == 3 and fluent[0] == "At" and fluent[1] == obj:
            return fluent[2]
    return None


def _bfs_path(start, goal, adjacency: dict) -> list:
    """Encuentra una ruta corta entre dos celdas usando BFS."""
    if start == goal:
        return [start]

    frontier = deque([[start]])
    visited = {start}

    while frontier:
        path = frontier.popleft()
        current = path[-1]

        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue

            new_path = path + [neighbor]

            if neighbor == goal:
                return new_path

            visited.add(neighbor)
            frontier.append(new_path)

    return []


def _build_move_action(robot, from_cell, to_cell) -> Action:
    """Construye una acción Move aterrizada."""
    return _make_action(
        name="Move",
        args=(robot, from_cell, to_cell),
        precond_pos=[
            ("At", robot, from_cell),
            ("Adjacent", from_cell, to_cell),
            ("Free", to_cell),
        ],
        precond_neg=[],
        add_list=[
            ("At", robot, to_cell),
            ("Free", from_cell),
        ],
        del_list=[
            ("At", robot, from_cell),
            ("Free", to_cell),
        ],
    )


def _build_pickup_action(robot, obj, loc) -> Action:
    """Construye una acción PickUp aterrizada."""
    return _make_action(
        name="PickUp",
        args=(robot, obj, loc),
        precond_pos=[
            ("At", robot, loc),
            ("At", obj, loc),
            ("HandsFree", robot),
            ("Pickable", obj),
        ],
        precond_neg=[],
        add_list=[
            ("Holding", robot, obj),
        ],
        del_list=[
            ("At", obj, loc),
            ("HandsFree", robot),
        ],
    )


def _build_putdown_action(robot, obj, loc) -> Action:
    """Construye una acción PutDown aterrizada."""
    return _make_action(
        name="PutDown",
        args=(robot, obj, loc),
        precond_pos=[
            ("At", robot, loc),
            ("Holding", robot, obj),
        ],
        precond_neg=[],
        add_list=[
            ("At", obj, loc),
            ("HandsFree", robot),
        ],
        del_list=[
            ("Holding", robot, obj),
        ],
    )


def _build_setup_supplies_action(robot, supplies, loc) -> Action:
    """Construye una acción SetupSupplies aterrizada."""
    return _make_action(
        name="SetupSupplies",
        args=(robot, supplies, loc),
        precond_pos=[
            ("At", robot, loc),
            ("MedicalPost", loc),
            ("Holding", robot, supplies),
        ],
        precond_neg=[],
        add_list=[
            ("SuppliesReady", loc),
            ("HandsFree", robot),
        ],
        del_list=[
            ("Holding", robot, supplies),
        ],
    )


def _build_rescue_action(robot, patient, loc) -> Action:
    """Construye una acción Rescue aterrizada."""
    return _make_action(
        name="Rescue",
        args=(robot, patient, loc),
        precond_pos=[
            ("At", robot, loc),
            ("At", patient, loc),
            ("MedicalPost", loc),
            ("SuppliesReady", loc),
        ],
        precond_neg=[],
        add_list=[
            ("Rescued", patient),
        ],
        del_list=[
            ("At", patient, loc),
        ],
    )


def _build_navigate_hla(robot, start, goal, adjacency: dict) -> HLA:
    """
    Construye una HLA Navigate(start, goal).

    Para mantener la jerarquía simple, se refina directamente en una secuencia
    de acciones Move siguiendo una ruta BFS.
    """
    hla = HLA(f"Navigate({start},{goal})")

    path = _bfs_path(start, goal, adjacency)

    if not path:
        hla.refinements = []
        return hla

    moves = []

    for i in range(len(path) - 1):
        moves.append(_build_move_action(robot, path[i], path[i + 1]))

    hla.refinements = [moves]
    return hla


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    ### Your code here ###
    # -----------------------------------------------------------------------
    #
    # Primera versión
    #
    # for hla in hlas:
    #     for refinement in hla.refinements:
    #         if is_plan_primitive(refinement):
    #             return refinement
    # return []
    #
    # Problema detectado en la primera versión:
    # Esta primera idea solo revisaba los refinamientos directos de las HLAs.
    # El error es que una HLA puede refinarse en una mezcla de acciones
    # primitivas y otras HLAs. Entonces no basta con mirar un solo nivel.
    # Además, tampoco se estaba simulando el plan para verificar si realmente
    # alcanzaba el objetivo del problema.
    #
    # Prompt utilizado:
    # "Tengo esta primera versión de hierarchicalSearch pero creo que solo
    # estoy mirando el primer refinamiento y no entiendo bien cómo seguir
    # refinando si dentro del plan todavía quedan HLAs me puedes ayudar a
    # hacerlo con BFS y a revisar al final si el plan sí cumple el objetivo?"
    #
    # -----------------------------------------------------------------------

    if not hlas:
        return []

    frontier = deque()
    start_plan: list[Action | HLA] = list(hlas)
    frontier.append(start_plan)

    visited = {_plan_key(start_plan)}

    while frontier:
        current_plan = frontier.popleft()

        if is_plan_primitive(current_plan):
            primitive_plan = current_plan

            if _simulate_plan(problem, primitive_plan):
                return primitive_plan

            continue

        hla_index = _find_first_hla_index(current_plan)

        if hla_index is None:
            continue

        current_hla = current_plan[hla_index]

        for refinement in current_hla.refinements:
            new_plan = (
                current_plan[:hla_index]
                + refinement
                + current_plan[hla_index + 1:]
            )

            key = _plan_key(new_plan)

            if key in visited:
                continue

            visited.add(key)
            frontier.append(new_plan)

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    ### Your code here ###
    # -----------------------------------------------------------------------   
    #
    # Primera versión:
    #
    # mission = HLA("FullRescueMission")
    # mission.refinements = [
    #     [
    #         HLA("PrepareSupplies"),
    #         HLA("ExtractPatient"),
    #     ]
    # ]
    # return [mission]
    #
    # Problema detectado en la primera versión:
    # Esta primera idea estaba demasiado incompleta porque creaba HLAs con
    # nombres generales, pero no las conectaba con acciones reales del problema.
    # Faltaban los objetos concretos del mapa, como el paciente, los suministros,
    # el puesto médico y las celdas. Además, Navigate no tenía movimientos Move
    # reales, entonces el planificador jerárquico no podía terminar en un plan
    # completamente primitivo.
    #
    # Prompt utilizado:
    # "Tengo esta idea de build_htn_hierarchy con FullRescueMission,
    # PrepareSupplies y ExtractPatient, pero esta muy general y no se como
    # conectarla con las acciones reales del mapa. me puedes ayudar a sacar
    # del estado inicial el robot, pacientes, suministros, puesto médico y
    # adyacencias para armar HLAs que terminen en acciones primitivas?"
    # -----------------------------------------------------------------------

    state = _initial_state(problem)
    goal = _goal_state(problem)

    at_fluents = _extract_fluents(state, "At")
    adjacent_fluents = _extract_fluents(state, "Adjacent")
    medical_posts = [fluent[1] for fluent in _extract_fluents(state, "MedicalPost")]
    pickables = [fluent[1] for fluent in _extract_fluents(state, "Pickable")]

    adjacency: dict = {}

    for fluent in adjacent_fluents:
        if len(fluent) != 3:
            continue

        _, cell_a, cell_b = fluent
        adjacency.setdefault(cell_a, []).append(cell_b)

    robot = None

    for fluent in at_fluents:
        if len(fluent) == 3 and str(fluent[1]).lower().startswith("robot"):
            robot = fluent[1]
            break

    if robot is None:
        robot = "robot"

    robot_location = _find_location(state, robot)

    if robot_location is None:
        return []

    patients = []

    if goal is not None:
        for fluent in goal:
            if len(fluent) == 2 and fluent[0] == "Rescued":
                patients.append(fluent[1])

    if not patients:
        for obj in pickables:
            if str(obj).lower().startswith("patient"):
                patients.append(obj)

    supplies = [
        obj for obj in pickables
        if str(obj).lower().startswith("supplies")
    ]

    if not supplies:
        supplies = [
            obj for obj in pickables
            if obj not in patients
        ]

    if not patients or not supplies or not medical_posts:
        return []

    current_robot_location = robot_location
    top_level_missions: list[HLA] = []

    for index, patient in enumerate(patients):
        supplies_index = min(index, len(supplies) - 1)
        current_supplies = supplies[supplies_index]
        medical_post = medical_posts[0]

        supplies_location = _find_location(state, current_supplies)
        patient_location = _find_location(state, patient)

        if supplies_location is None or patient_location is None:
            continue

        navigate_to_supplies = _build_navigate_hla(
            robot,
            current_robot_location,
            supplies_location,
            adjacency,
        )

        navigate_supplies_to_post = _build_navigate_hla(
            robot,
            supplies_location,
            medical_post,
            adjacency,
        )

        prepare_supplies = HLA(
            f"PrepareSupplies({current_supplies},{medical_post})",
            refinements=[
                [
                    navigate_to_supplies,
                    _build_pickup_action(robot, current_supplies, supplies_location),
                    navigate_supplies_to_post,
                    _build_setup_supplies_action(robot, current_supplies, medical_post),
                ]
            ],
        )

        navigate_to_patient = _build_navigate_hla(
            robot,
            medical_post,
            patient_location,
            adjacency,
        )

        navigate_patient_to_post = _build_navigate_hla(
            robot,
            patient_location,
            medical_post,
            adjacency,
        )

        extract_patient = HLA(
            f"ExtractPatient({patient},{medical_post})",
            refinements=[
                [
                    navigate_to_patient,
                    _build_pickup_action(robot, patient, patient_location),
                    navigate_patient_to_post,
                    _build_putdown_action(robot, patient, medical_post),
                ]
            ],
        )

        full_rescue_mission = HLA(
            f"FullRescueMission({current_supplies},{patient},{medical_post})",
            refinements=[
                [
                    prepare_supplies,
                    extract_patient,
                    _build_rescue_action(robot, patient, medical_post),
                ]
            ],
        )

        top_level_missions.append(full_rescue_mission)
        current_robot_location = medical_post

    return top_level_missions
    ### End of your code ###