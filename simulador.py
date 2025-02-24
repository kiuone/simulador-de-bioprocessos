import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Definição dos modelos cinéticos
def monod(S):
    return u_max * (S / (Ks + S))

def andrews(S):
    return u_max * (S / (Ks + S + (S**2/Kis)))

def levenspiel(S, P):
    return u_max * (S/(Ks + S)) * (1 - (P/Cp_star)**L)

def lee_pollard_coulman(S, X):
    return u_max * (S/(Ks + S)) * (1 - (X/Cx_star)**M)

# Função para resolver as EDOs
def sistema_edo(y, t, modelo, morte_celular):
    X, S, P = y
    if modelo == "Monod":
        u = monod(S)
    elif modelo == "Andrews":
        u = andrews(S)
    elif modelo == "Levenspiel":
        u = levenspiel(S, P)
    elif modelo == "Lee-Pollard-Coulman":
        u = lee_pollard_coulman(S, X)

    if morte_celular:
        dXdt = (u - Kd) * X
    else:
        dXdt = u * X

    dSdt = (-1/Ys) * u * X - m * X
    dPdt = (Yp/Ys) * u * X

    return [dXdt, dSdt, dPdt]

# Simulação do modelo
def simular_modelo(modelo, tempo, morte_celular):
    solucao = odeint(sistema_edo, y0, tempo, args=(modelo, morte_celular))
    return solucao

# Cálculo do erro quadrático médio
def calcular_erro(dados_simulados, dados_experimentais):
    return np.mean((dados_simulados - dados_experimentais)**2)

# Interface no Streamlit
st.title("Simulador de Processos Industriais")

# Entrada dos parâmetros
u_max = st.number_input("u_max: Taxa máxima de crescimento (1/min)", 0.001, 1.0, 0.0153)
Ks = st.number_input("Ks: Constante de saturação (g/L)", 1.0, 500.0, 165.0)
Kd = st.number_input("Kd: Taxa de morte celular (g/L)", 0.0, 0.1, 0.001)
Kis = st.number_input("Kis: Constante de inibição (g/L)", 1.0, 200.0, 50.0)
L = st.number_input("Largura do reator: Parâmetro L", 0.1, 2.0, 0.5)
M = st.number_input("Parâmetro M", 0.1, 2.0, 0.4)
Cp_star = st.number_input("Concentração inibidora do produto (g/L)", 10.0, 200.0, 80.0)
Cx_star = st.number_input("Concentração máxima de células (g/mL)", 1.0, 10.0, 6.0)
m = st.number_input("Coeficiente de manutenção (g)", 0.001, 0.1, 0.03)
Ys = st.number_input("Rendimento de biomassa por substrato", 0.001, 1.0, 0.0087)
Yp = st.number_input("Rendimento do produto por substrato", 0.001, 1.0, 0.44)

# Entrada dos dados experimentais
st.subheader("Insira os dados experimentais do reator")

tempo = st.text_area("Tempo (h)", "0,10,25,30,50,55,70,80,140,160,200,250,300")
celulas = st.text_area("Biomassa (g/L)", "2.4,2.42,2.43,2.5,2.8,2.9,3.1,3.2,3.2,3.1,3.0,2.8,2.7")
substrato = st.text_area("Substrato (g/L)", "272,250,238,206,188,177,160,140,128,118,115,110,105")
produto = st.text_area("Produto (g/L)", "0,10,23,29,35,43,48,60,65,67,70,77,79")

# Processar entrada do usuário
try:
    tempo = np.array([float(x) for x in tempo.split(",")])
    celulas = np.array([float(x) for x in celulas.split(",")])
    substrato = np.array([float(x) for x in substrato.split(",")])
    produto = np.array([float(x) for x in produto.split(",")])

    # Condições iniciais
    y0 = [celulas[0], substrato[0], produto[0]]

    # Modelos disponíveis
    modelos = ["Monod", "Andrews", "Levenspiel", "Lee-Pollard-Coulman"]
    erros = {}
    resultados = {}

    # Simular e calcular erro para cada modelo
    for modelo in modelos:
        resultado_com_morte = simular_modelo(modelo, tempo, morte_celular=True)
        resultado_sem_morte = simular_modelo(modelo, tempo, morte_celular=False)

        resultados[f"{modelo} - Com Morte Celular"] = resultado_com_morte
        resultados[f"{modelo} - Sem Morte Celular"] = resultado_sem_morte

        erro_com_morte = calcular_erro(resultado_com_morte[:, 0], celulas)
        erro_sem_morte = calcular_erro(resultado_sem_morte[:, 0], celulas)

        erros[f"{modelo} - Com Morte Celular"] = erro_com_morte
        erros[f"{modelo} - Sem Morte Celular"] = erro_sem_morte

    # Melhor modelo (menor erro)
    melhor_modelo = min(erros, key=erros.get)

    # Exibir resultados
    st.subheader("Resultados")
    st.write("Erro para cada modelo:")
    for modelo, erro in erros.items():
        st.write(f"{modelo}: {erro:.4f}")

    # st.success(f"O melhor modelo para os dados fornecidos é: **{melhor_modelo}**")

except Exception as e:
    st.error("Erro ao processar os dados. Verifique a formatação.")

# Estilização dos gráficos
sns.set_theme(style="whitegrid")

# Criar dicionários para armazenar R² e RMSE
r2_scores = {}
rmse_scores = {}

# Calcular métricas para cada modelo
for nome, resultado in resultados.items():
    r2_scores[nome] = {
        "Células": r2_score(celulas, resultado[:, 0]),
        "Substrato": r2_score(substrato, resultado[:, 1]),
        "Produto": r2_score(produto, resultado[:, 2])
    }
    rmse_scores[nome] = {
        "Células": np.sqrt(mean_squared_error(celulas, resultado[:, 0])),
        "Substrato": np.sqrt(mean_squared_error(substrato, resultado[:, 1])),
        "Produto": np.sqrt(mean_squared_error(produto, resultado[:, 2]))
    }

# Gráfico principal de comparação entre os modelos
fig1, axs = plt.subplots(3, 1, figsize=(15, 12))

# Gráfico para concentração celular
axs[0].scatter(tempo, celulas, color='black', label='Experimental', zorder=3)
for nome, resultado in resultados.items():
    axs[0].plot(tempo, resultado[:, 0], '--', label=f"{nome} (R²={r2_scores[nome]['Células']:.3f}, RMSE={rmse_scores[nome]['Células']:.3f})")
axs[0].set_xlabel('Tempo (h)')
axs[0].set_ylabel('Células (g/L)')
axs[0].set_title('Concentração Celular')
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[0].grid(True)

# Gráfico para concentração de substrato
axs[1].scatter(tempo, substrato, color='black', label='Experimental', zorder=3)
for nome, resultado in resultados.items():
    axs[1].plot(tempo, resultado[:, 1], '--', label=f"{nome} (R²={r2_scores[nome]['Substrato']:.3f}, RMSE={rmse_scores[nome]['Substrato']:.3f})")
axs[1].set_xlabel('Tempo (h)')
axs[1].set_ylabel('Substrato (g/L)')
axs[1].set_title('Concentração de Substrato')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].grid(True)

# Gráfico para concentração de produto
axs[2].scatter(tempo, produto, color='black', label='Experimental', zorder=3)
for nome, resultado in resultados.items():
    axs[2].plot(tempo, resultado[:, 2], '--', label=f"{nome} (R²={r2_scores[nome]['Produto']:.3f}, RMSE={rmse_scores[nome]['Produto']:.3f})")
axs[2].set_xlabel('Tempo (h)')
axs[2].set_ylabel('Produto (g/L)')
axs[2].set_title('Concentração de Produto')
axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[2].grid(True)

plt.tight_layout()
st.pyplot(fig1)  # Exibir primeiro gráfico

# Criar um segundo gráfico para comparar modelos com e sem morte celular
modelo_escolhido = melhor_modelo.split(" - ")[0]  # Pega o nome base do melhor modelo
if f"{modelo_escolhido} - Com Morte Celular" in resultados and f"{modelo_escolhido} - Sem Morte Celular" in resultados:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    ax2.plot(tempo, resultados[f"{modelo_escolhido} - Com Morte Celular"][:, 0], 'r-', label='Com Morte Celular')
    ax2.plot(tempo, resultados[f"{modelo_escolhido} - Sem Morte Celular"][:, 0], 'b--', label='Sem Morte Celular')
    ax2.scatter(tempo, celulas, color='black', label='Experimental', zorder=3)
    
    ax2.set_xlabel('Tempo (h)')
    ax2.set_ylabel('Células (g/L)')
    ax2.set_title(f'Comparação entre Modelos ({modelo_escolhido})')
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)  # Exibir segundo gráfico

# Exibir o melhor modelo
st.success(f"\n🔹 **Melhor modelo:** {melhor_modelo} com RMSE total de {erros[melhor_modelo]:.6f}")
