import itertools
import pandas as pd
import matplotlib.pyplot as plt

# Dados fornecidos
apps = ['Sudoku', 'Convolution', 'Inference']
devices = ['CPU', 'GPU', 'NPU']

execution_times = {
    'Sudoku': {'CPU': 0.5318, 'GPU': 11.18, 'NPU': 10.713},
    'Convolution': {'CPU': 608.34, 'GPU': 0.8732, 'NPU': 225},
    'Inference': {'CPU': 304.40, 'GPU': 151.598, 'NPU': 6.685},
}

power = {'CPU': 65, 'GPU': 160, 'NPU': 5}
cost = {'CPU': 140, 'GPU': 400, 'NPU': 160}

results = []

# Todas as combinações possíveis de dispositivos para as aplicações
device_combinations = list(itertools.product(devices, repeat=3))

for combo in device_combinations:
    time_seq = 0
    energy_seq = 0
    edp_seq = 0
    devices_used = set(combo)

    # Execução sequencial (com dependência de dados)
    for i, app in enumerate(apps):
        dev = combo[i]
        t = execution_times[app][dev] / 1000
        p = power[dev]
        energy = t * p
        edp = energy * t
        time_seq += t
        energy_seq += energy
        edp_seq += edp

    cost_seq = sum(cost[dev] for dev in devices_used)
    cost_perf_seq = cost_seq * time_seq
    cost_energy_seq = cost_seq * energy_seq

    # Execução concorrente (sem dependência de dados)
    times_conc = [execution_times[app][combo[i]] / 1000 for i, app in enumerate(apps)]
    energies_conc = [execution_times[app][combo[i]] / 1000 * power[combo[i]] for i, app in enumerate(apps)]
    edps_conc = [e * t for e, t in zip(energies_conc, times_conc)]
    time_conc = max(times_conc)
    energy_conc = sum(energies_conc)
    edp_conc = sum(edps_conc)
    cost_conc = sum(cost[dev] for dev in set(combo))
    cost_perf_conc = cost_conc * time_conc
    cost_energy_conc = cost_conc * energy_conc

    results.append({
        'Sudoku': combo[0],
        'Convolution': combo[1],
        'Inference': combo[2],
        'Execução': 'Sequencial',
        'Tempo (s)': time_seq,
        'Energia (J)': energy_seq,
        'EDP': edp_seq,
        'Custo ($)': cost_seq,
        'Custo x Tempo ($.s)': cost_perf_seq,
        'Custo x Energia ($.J)': cost_energy_seq
    })

    results.append({
        'Sudoku': combo[0],
        'Convolution': combo[1],
        'Inference': combo[2],
        'Execução': 'Concorrente',
        'Tempo (s)': time_conc,
        'Energia (J)': energy_conc,
        'EDP': edp_conc,
        'Custo ($)': cost_conc,
        'Custo x Tempo ($.s)': cost_perf_conc,
        'Custo x Energia ($.J)': cost_energy_conc
    })

# Converter em DataFrame
df_results = pd.DataFrame(results)

# Filtra resultados por tipo de execução
df_seq = df_results[df_results['Execução'] == 'Sequencial']
df_conc = df_results[df_results['Execução'] == 'Concorrente']

# Identificar os melhores resultados para execução sequencial
best_seq_time = df_seq.loc[df_seq['Tempo (s)'].idxmin()]
best_seq_energy = df_seq.loc[df_seq['Energia (J)'].idxmin()]
best_seq_edp = df_seq.loc[df_seq['EDP'].idxmin()]
best_seq_cost_time = df_seq.loc[df_seq['Custo x Tempo ($.s)'].idxmin()]
best_seq_cost_energy = df_seq.loc[df_seq['Custo x Energia ($.J)'].idxmin()]

# Identificar os melhores resultados para execução concorrente
best_conc_time = df_conc.loc[df_conc['Tempo (s)'].idxmin()]
best_conc_energy = df_conc.loc[df_conc['Energia (J)'].idxmin()]
best_conc_edp = df_conc.loc[df_conc['EDP'].idxmin()]
best_conc_cost_time = df_conc.loc[df_conc['Custo x Tempo ($.s)'].idxmin()]
best_conc_cost_energy = df_conc.loc[df_conc['Custo x Energia ($.J)'].idxmin()]

# Tabela com os melhores por critério
melhores_seq = pd.DataFrame([best_seq_time, best_seq_energy, best_seq_edp, best_seq_cost_time, best_seq_cost_energy])
melhores_seq['Critério'] = ['Menor Tempo', 'Menor Energia', 'Menor EDP', 'Melhor Custo-Tempo', 'Melhor Custo-Energia']

# Tabela com os melhores por critério
melhores_conc = pd.DataFrame([best_conc_time, best_conc_energy, best_conc_edp, best_conc_cost_time, best_conc_cost_energy])
melhores_conc['Critério'] = ['Menor Tempo', 'Menor Energia', 'Menor EDP', 'Melhor Custo-Tempo', 'Melhor Custo-Energia']

# Exibir todas as combinações (caso queira salvar ou visualizar)
print("Resultados completos:")
print(df_results)

# Exibir os melhores
print("\nMelhores resultados por critério (SEQUENCIAL):")
print(melhores_seq)

print("\nMelhores resultados por critério (CONCORRENTE):")
print(melhores_conc)

# Gráfico 1: Tempo x Custo
plt.figure(figsize=(10, 6))
for exec_type in ['Sequencial', 'Concorrente']:
    subset = df_results[df_results['Execução'] == exec_type]
    plt.scatter(subset['Tempo (s)'], subset['Custo ($)'], label=exec_type, alpha=0.7)

plt.title('DSE: Tempo (s) vs Custo ($)')
plt.xlabel('Tempo (s)', fontsize=14)
plt.ylabel('Custo ($)', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig("grafico_dse_tempo_custo.png")

# Gráfico 2: Custo x Energia
plt.figure(figsize=(10, 6))
for exec_type in ['Sequencial', 'Concorrente']:
    subset = df_results[df_results['Execução'] == exec_type]
    plt.scatter(subset['Custo ($)'], subset['Energia (J)'], label=exec_type, alpha=0.7)

plt.title('DSE: Custo ($) vs Energia (J)')
plt.xlabel('Custo ($)', fontsize=14)
plt.ylabel('Energia (J)', fontsize=14)
plt.grid(True)
plt.savefig("grafico_dse_custo_energia.png")


# Gráfico 3: 
plt.figure(figsize=(10, 6))
for exec_type in ['Sequencial', 'Concorrente']:
    subset = df_results[df_results['Execução'] == exec_type]
    plt.scatter(subset['Custo x Tempo ($.s)'], subset['Custo x Energia ($.J)'], label=exec_type, alpha=0.7)

plt.title('DSE: Custo x Tempo vs Custo x Energia')
plt.xlabel('Custo x Tempo ($.s)', fontsize=14)
plt.ylabel('Custo x Energia ($.J)', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig("grafico_dse_custoXtempo_custoXenergia.png")