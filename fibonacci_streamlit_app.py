import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
# Codigo de la optimizacion
def leer_funcion(funcion_str: str):
    patron_exponencial = r"e\^\(([x0-9\+\-\*/\.\(\)\s]+)\)"
    funcion_str = re.sub(patron_exponencial, r"np.exp(\1)", funcion_str)
    patron_exponencial_simple = r"e\^([-+]?[\d\.]*\*?x)"
    funcion_str = re.sub(patron_exponencial_simple, r"np.exp(\1)", funcion_str)
    funcion_str = funcion_str.replace("ln", "np.log")
    funcion_str = funcion_str.replace("log", "np.log10")
    funcion_str = funcion_str.replace("sin", "np.sin")
    funcion_str = funcion_str.replace("cos", "np.cos")
    funcion_str = funcion_str.replace("tan", "np.tan")
    funcion_str = funcion_str.replace("^", "**")
    funcion_str = re.sub(r"(?<=[0-9])(?=[0-9])", "*", funcion_str)
    funcion_str = re.sub(r"(?<=[0-9])(?=[a-zA-Z])", "*", funcion_str)

    return eval("lambda x: " + funcion_str)


def verificar_funcion(funcion, a, b, puntos=100, max_mostrar=5):
    try:
        x_vals = np.linspace(a, b, puntos)
        x_vals = np.array(x_vals)
        y_vals = funcion(x_vals)

        nans = np.isnan(y_vals)
        inftys = np.isinf(y_vals)

        problematic_x = x_vals[nans | inftys]
        no_def = len(problematic_x)

        if no_def > 0:
            st.error("La función tiene problemas de dominio en el intervalo.")
            mostrar = min(no_def, max_mostrar)
            for i in range(mostrar):
                x = problematic_x[i]
                try:
                    y = funcion(x)
                except:
                    y = "Error"
                st.write(f"f({x:.6f}) = {y}")
            if no_def > max_mostrar:
                st.write(f"... y {no_def - max_mostrar} valores problemáticos más.")
            return False

        return True

    except Exception as e:
        st.error(f"Error al evaluar la función: {e}")
        return False

def metodo_fibonacci(funcion, a, b, max_iter):
    fib = [0, 1]
    for _ in range(2, max_iter + 1):
        fib.append(fib[-1] + fib[-2])

    n = max_iter
    pasos = []

    L = b - a
    x1 = a + (fib[n - 2] / fib[n]) * L
    x2 = a + (fib[n - 1] / fib[n]) * L
    try:
        f1 = funcion(x1)
    except:
        f1 = np.inf
    try:
        f2 = funcion(x2)
    except:
        f2 = np.inf

    for i in range(1, n):
        pasos.append((i, x1, f1, x2, f2, a, b))

        if np.isclose(x1, x2, rtol=1e-5, atol=1e-5):
            a1, b1 = a, x1
            a2, b2 = x2, b
            minimo1 = (a1 + b1) / 2
            minimo2 = (a2 + b2) / 2
            return pasos, [(minimo1, funcion(minimo1)), (minimo2, funcion(minimo2))]

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fib[n - i - 2] / fib[n - i]) * (b - a)
            f1 = funcion(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fib[n - i - 1] / fib[n - i]) * (b - a)
            f2 = funcion(x2)

    minimo = (a + b) / 2
    return pasos, [(minimo, funcion(minimo))]

# Interfaz  
st.title("Método de Optimización por Fibonacci")

with st.sidebar:
    st.header("Configuración")
    modo = st.radio("¿Qué deseas hacer?", ["Minimizar", "Maximizar"])
    funcion_str = st.text_input("Función (usa 'np.')", "(x - 2)**2")
    a = st.number_input("Extremo izquierdo (a)", value=0.0)
    b = st.number_input("Extremo derecho (b)", value=5.0)
    max_iter = st.slider("Número de iteraciones", 3, 20, value=6)
    ejecutar = st.button("Ejecutar algoritmo")

if ejecutar:
    funcion = leer_funcion(funcion_str)

    if verificar_funcion(funcion, a, b):
        if modo == "Maximizar":
            funcion_original = funcion
            funcion = lambda x: -funcion_original(x)
        pasos, minimos = metodo_fibonacci(funcion, a, b, max_iter)
        st.code(f"Función: {funcion_str}")
        st.write(f"Intervalo: [{a:.2f}, {b:.2f}]")
        st.write(f"Iteraciones: {max_iter}")
        

        st.subheader("Iteraciones")
        for i, x1, f1, x2, f2, ai, bi in pasos:
            st.write(f"Iteración {i}: x1 = {x1:.6f}, f(x1) = {f1:.6f} | x2 = {x2:.6f}, f(x2) = {f2:.6f}")
            st.write(f"   Intervalo: [{ai:.6f}, {bi:.6f}]")

        st.subheader("Resultados")
        if modo == "Minimizar":
            for idx, (x, fx) in enumerate(minimos, 1):
                st.success(f"Mínimo {idx}: x = {x:.6f}, f(x) = {fx:.6f}")
        else:
            for idx, (x, fx) in enumerate(minimos, 1):
                st.success(f"Máximo {idx}: x = {x:.6f}, f(x) = {-fx:.6f}")

        # Gráfica de la función
        st.subheader("Visualización")
        x_vals = np.linspace(a, b, 300)
        y_vals = funcion(x_vals)
        if modo == "Maximizar":
            y_vals = -y_vals
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label='f(x)')
        etiqueta = "mín" if modo == "Minimizar" else "máx"
        for x, fx in minimos:
            if modo == "Maximizar":
                fx = -fx
            ax.plot(x, fx, 'ro')
            ax.annotate(f"{etiqueta} ({x:.2f}, {fx:.2f})", (x, fx), textcoords="offset points", xytext=(0,10), ha='center')
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Función y puntos {etiqueta}imo")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("Corrige la función o ajusta el intervalo para continuar.")
