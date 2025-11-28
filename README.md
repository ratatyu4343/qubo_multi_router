# PCB Routing Experiments

Набір інструментів для порівняння алгоритмів трасування (класичні Beam/Wave та квантовий MultiNet Quantum Router з генератором кандидатів A*). Є утиліти для генерації випадкових плат, запуску серій експериментів, збору метрик та візуалізації.

## Залежності
- Python 3.12+
- Встановити: `pip install -r requirements.txt`
- Опційно для реального D-Wave (`use_dwave=True`): `dwave-system`

> Примітка: у режимі `use_dwave=False` квантовий роутер працює на класичній евристиці (simulated annealing через `neal`) і **не** використовує реальне квантове залізо. Прапорець `use_dwave=True` наразі заглушка — інтеграцію з реальним D-Wave у цьому репозиторії не реалізовано.

## Структура
- `pcb_routing/` — роутери, генератори кандидатів (A*), метрики, візуалізація.
- `experiments/` — обгортка для конфігів, метрик/зображень/маніфестів.
- `saved_experiments/` — збережені запуски для аналізу без повторного прогону.
- `tests/` — smoke- та інтеграційні тести.
- `main.py` — готові сценарії (див. нижче).

## Швидкий старт
1. Встановіть залежності: `pip install -r requirements.txt`
2. Перевірте тести: `pytest`
3. Запустіть потрібний сценарій (див. приклади).

## Приклади використання (`main.py`)

### 1) Генерація і запуск 1000 випадкових кейсів 20x20
```python
def main() -> None:
    expeinets_boards = []
    for i in range(1000):
        width = 20
        height = 20
        num_pins = random.randint(2, 20)
        num_nets = random.randint(2, 30)
        expeinets_boards.append(Experiment.generate_random_case(width, height, num_pins, num_nets))
    
    for experiment_id, board_nets in enumerate(expeinets_boards):
        experiment = Experiment(
            name=f"compare_routers_20x20_{experiment_id}",
            board=board_nets[0],
            nets=board_nets[1],
            output_dir=Path("saved_experiments2"),
        )
        register_configurations(experiment)
        experiment.run_all(save_images=True)
        print(f"Completed experiment {experiment_id}")
```
Запуск (у `__main__` активний `main2`, тому викликайте явно):  
`python -c "import main; main.main()"`  
Артефакти: `saved_experiments2/compare_routers_20x20_*`.

### 2) Порівняння на готових метриках (`saved_experiments2`)
Активний за замовчуванням `main3()`:
```python
def main3() -> None: 
    df = Experiment.load_all_metrics('saved_experiments2')
    print('Beam vs Wave by Routed Count (%):', compare_two_configs_by_routed_count("classical_beam", "classical_wave", df))
    print('Quantum A* vs Beam by Routed Count (%):', compare_two_configs_by_routed_count("classical_beam", "quantum_astar", df))
    print('Quantum A* vs Wave by Routed Count (%):', compare_two_configs_by_routed_count("classical_wave", "quantum_astar", df))

    print('Beam vs Wave by Length Count (%):', compare_two_configs_by_length("classical_beam", "classical_wave", df))
    print('Quantum A* vs Beam by Length Count (%):', compare_two_configs_by_length("classical_beam", "quantum_astar", df))
    print('Quantum A* vs Wave by Length Count (%):', compare_two_configs_by_length("classical_wave", "quantum_astar", df))
```
Запуск: `python main.py` (або `python -m main`).

### 3) Порівняння на іншому каталозі метрик (`saved_experiments`)
```python
def main2() -> None:
    df = Experiment.load_all_metrics('saved_experiments')
    print("Beam vs Wave by Routed Count (%):", compare_two_configs_by_routed_count("classical_beam", "classical_wave", df))
    print("Quantum A* vs Beam by Routed Count (%):", compare_two_configs_by_routed_count("classical_beam", "quantum_astar", df))
    print("Quantum A* vs Wave by Routed Count (%):", compare_two_configs_by_routed_count("classical_wave", "quantum_astar", df))

    print("Beam vs Wave by Length (%):", compare_two_configs_by_length("classical_beam", "classical_wave", df))
    print("Quantum A* vs Beam by Length (%):", compare_two_configs_by_length("classical_beam", "quantum_astar", df))
    print("Quantum A* vs Wave by Length (%):", compare_two_configs_by_length("classical_wave", "quantum_astar", df))
```
Запуск: `python -c "import main; main.main2()"`.

## Конфігурації роутерів (`register_configurations`)
- `classical_beam` — Beam Router (1 шар).
- `classical_wave` — Wave Router (1 шар).
- `quantum_astar` — MultiNet Quantum Router з кандидатами A* (`per_net_limit=30`, 1 шар, `use_dwave=False` за замовчуванням).

## Тестування
- Повний прогін: `pytest`
- Окремий тест: `pytest tests/test_experiment_integration.py::test_experiment_run_all_creates_outputs`

## Підказки
- Артефакти лежать у `<output_dir>/<experiment_name>/` з підкаталогами `metrics/`, `solutions/`, `images/`, `metrics_log.csv`.
- Для швидких smoke-сценаріїв використовуйте малі сітки (3x3, 5x5) і мало нетів.
- Якщо додаєте нові генератори, зареєструйте їх у `pcb_routing/candidates/__init__.py` або через `MultiNetQuantumRouter.register_generator`.
