import os
from pathlib import Path

from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from google.adk.agents import Agent

# Carga las credenciales del archivo .env (está en esta misma carpeta)
load_dotenv(Path(__file__).parent / ".env")

# Lee los valores del .env
proveedor = os.getenv("PROVEEDOR", "gemini")  # gemini, openai u ollama
clave_openai = os.getenv("OPEN_API_KEY")
clave_ollama = os.getenv("OLLAMA_API_KEY")

if proveedor == "openai":
    from google.adk.models.lite_llm import LiteLlm

    modelo = LiteLlm(model="openai/gpt-5.6-luna", api_key=clave_openai)
elif proveedor == "ollama":
    from google.adk.models.lite_llm import LiteLlm

    # Ollama usa un formato compatible con OpenAI,
    # por eso el modelo empieza con "openai/"
    modelo = LiteLlm(
        model="openai/gemma4:31b",
        api_base="https://ollama.com/v1",  # dirección de la nube de Ollama
        api_key=clave_ollama,
    )
else:
    # Gemini lee la clave GEMINI_API_KEY que cargó load_dotenv
    modelo = "gemini-flash-lite-latest"

# Carpeta donde están los archivos CSV (junto a este archivo)
carpeta = Path(__file__).parent / "datos"


def _leer_csv(nombre_archivo: str) -> pd.DataFrame:
    ruta = carpeta / nombre_archivo
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV: {ruta}")
    return pd.read_csv(ruta)


def habitaciones_disponibles() -> dict:
    """Lista las habitaciones disponibles con su ID, distrito y universidad cercana."""
    habitaciones = _leer_csv("habitaciones.csv")
    disp = habitaciones[
        habitaciones["estado"].astype(str).str.strip().str.lower() == "disponible"
    ]

    lista_formateada = []
    for _, row in disp.iterrows():
        codigo = (
            str(row["codigo"]).strip()
            if "codigo" in row and pd.notna(row["codigo"])
            else ""
        )
        distrito = (
            str(row["distrito"]).strip()
            if "distrito" in row and pd.notna(row["distrito"])
            else ""
        )
        universidad = (
            str(row["universidad"]).strip()
            if "universidad" in row and pd.notna(row["universidad"])
            else "No especificada"
        )
        lista_formateada.append({
            "id": codigo,
            "distrito": distrito,
            "universidad_cercana": universidad,
        })

    return {
        "cantidad": len(lista_formateada),
        "habitaciones_disponibles": lista_formateada,
    }


def listar_habitaciones_resumen() -> dict:
    """Devuelve el resumen de habitaciones con código/ID, título, distrito, universidad cercana, precio y estado."""
    habitaciones = _leer_csv("habitaciones.csv")
    columnas = [
        col
        for col in ["codigo", "titulo", "distrito", "universidad", "precio", "estado"]
        if col in habitaciones.columns
    ]
    return {
        "total_habitaciones": int(len(habitaciones)),
        "habitaciones": habitaciones[columnas].fillna("").to_dict(orient="records"),
    }


def habitaciones_por_universidad(universidad: str = "") -> dict:
    """Filtra habitaciones según universidad cercana o lista las que tienen universidad registrada."""
    habitaciones = _leer_csv("habitaciones.csv")
    filtro_uni = habitaciones[
        habitaciones["universidad"].astype(str).str.strip().ne("")
        & habitaciones["universidad"].astype(str).str.lower().ne("nan")
    ]
    if universidad.strip():
        filtro_uni = filtro_uni[
            filtro_uni["universidad"]
            .astype(str)
            .str.contains(universidad.strip(), case=False, na=False)
        ]

    columnas = [
        col
        for col in ["codigo", "titulo", "distrito", "universidad", "precio", "estado"]
        if col in filtro_uni.columns
    ]
    return {
        "cantidad": int(len(filtro_uni)),
        "habitaciones": filtro_uni[columnas].fillna("").to_dict(orient="records"),
    }


def servicios_y_reglas_habitaciones(solo_disponibles: bool = True) -> dict:
    """Devuelve servicios, comodidades y reglas de convivencia de las habitaciones."""
    habitaciones = _leer_csv("habitaciones.csv")
    df = habitaciones
    if solo_disponibles and "estado" in df.columns:
        df = df[df["estado"].astype(str).str.strip().str.lower() == "disponible"]

    columnas = [
        col
        for col in [
            "codigo",
            "titulo",
            "distrito",
            "universidad",
            "precio",
            "estado",
            "bano_propio",
            "cocina",
            "condominio",
            "amoblado",
            "espacio",
            "ambiente",
            "visitas",
            "mascotas",
            "metodos_pago",
            "contrato",
            "reglas",
        ]
        if col in df.columns
    ]
    return {
        "cantidad": int(len(df)),
        "habitaciones": df[columnas].fillna("").to_dict(orient="records"),
    }


def listar_reservas() -> dict:
    """Devuelve todas las reservas registradas actualmente y su cantidad."""
    reservas = _leer_csv("reservas.csv")
    return {
        "cantidad_reservas": int(len(reservas)),
        "reservas": reservas.fillna("").to_dict(orient="records"),
    }


def buscar_persona(nombre: str) -> dict:
    """Busca estudiantes o anfitriones por nombre o parte del nombre."""
    estudiantes = _leer_csv("estudiantes.csv")
    anfitriones = _leer_csv("anfitriones.csv")

    est_encontrados = estudiantes[
        estudiantes["nombre"].astype(str).str.contains(nombre, case=False, na=False)
    ]
    anf_encontrados = anfitriones[
        anfitriones["nombre"].astype(str).str.contains(nombre, case=False, na=False)
    ]

    return {
        "estudiantes": est_encontrados.fillna("").to_dict(orient="records"),
        "anfitriones": anf_encontrados.fillna("").to_dict(orient="records"),
    }


def reservas_de_inquilino(nombre_inquilino: str) -> dict:
    """Devuelve las reservas de un inquilino o estudiante."""
    reservas = _leer_csv("reservas.csv")
    reservas_filtradas = reservas[
        reservas["inquilino"]
        .astype(str)
        .str.contains(nombre_inquilino, case=False, na=False)
    ]
    return {"reservas": reservas_filtradas.fillna("").to_dict(orient="records")}


def detalle_habitacion(codigo_habitacion: str) -> dict:
    """Devuelve toda la información de una habitación, su anfitrión y sus reservas a partir de su ID o código (ejemplo: 'HAB-001')."""
    habitaciones = _leer_csv("habitaciones.csv")
    reservas = _leer_csv("reservas.csv")
    anfitriones = _leer_csv("anfitriones.csv")

    cod = str(codigo_habitacion).strip().upper()
    hab = habitaciones[habitaciones["codigo"].astype(str).str.strip().str.upper() == cod]
    if hab.empty:
        return {"error": f"No se encontró ninguna habitación con el ID {cod}."}

    anf = anfitriones[
        anfitriones["codigo_habitacion"].astype(str).str.strip().str.upper() == cod
    ]
    res_hab = reservas[
        reservas["codigo_habitacion"].astype(str).str.strip().str.upper() == cod
    ]

    return {
        "id": cod,
        "habitacion": hab.fillna("").to_dict(orient="records")[0],
        "anfitrion": anf.fillna("").to_dict(orient="records"),
        "reservas": res_hab.fillna("").to_dict(orient="records"),
    }


def resumen_ingresos() -> dict:
    """Calcula el total acumulado en reservas y el promedio por reserva."""
    reservas = _leer_csv("reservas.csv")
    montos = pd.to_numeric(reservas["monto"], errors="coerce").dropna()
    total_ingresos = float(montos.sum()) if len(montos) > 0 else 0.0
    promedio_reserva = float(montos.mean()) if len(montos) > 0 else 0.0
    return {
        "cantidad_reservas": int(len(reservas)),
        "total_ingresos_soles": round(total_ingresos, 2),
        "promedio_reserva_soles": round(promedio_reserva, 2),
    }


def ingresos_por_metodo_pago() -> dict:
    """Suma los montos de reservas agrupados por método de pago para graficar."""
    reservas = _leer_csv("reservas.csv")
    df = reservas.copy()
    df["monto_num"] = pd.to_numeric(df["monto"], errors="coerce").fillna(0.0)
    total_por_metodo = df.groupby("metodo_pago", dropna=False)["monto_num"].sum()
    return {
        "etiquetas": [str(x) for x in total_por_metodo.index.tolist()],
        "valores": [round(float(v), 2) for v in total_por_metodo.tolist()],
    }


def reservas_por_distrito() -> dict:
    """Cuenta cuántas reservas hay por distrito para graficar."""
    reservas = _leer_csv("reservas.csv")
    conteo_distrito = reservas.groupby("distrito", dropna=False)["codigo_habitacion"].count()
    return {
        "etiquetas": [str(x) for x in conteo_distrito.index.tolist()],
        "valores": [int(v) for v in conteo_distrito.tolist()],
    }


def crear_grafico(
    titulo: str, tipo: str, etiquetas: list[str], valores: list[float]
) -> dict:
    """Muestra un gráfico al usuario. Úsala cuando pidan graficar algo.

    Args:
        titulo: título del gráfico.
        tipo: 'bar' (barras), 'line' (líneas) o 'pie' (torta).
        etiquetas: nombres de cada categoría.
        valores: número de cada categoría, en el mismo orden.
    """
    return {
        "titulo": titulo,
        "tipo": tipo,
        "etiquetas": etiquetas,
        "valores": valores,
    }


def registrar_reserva(
    codigo_habitacion: str,
    inquilino: str,
    metodo_pago: str,
    contrato: str,
    monto: float,
) -> dict:
    """Registra una reserva nueva en reservas.csv."""
    reservas = _leer_csv("reservas.csv")
    habitaciones = _leer_csv("habitaciones.csv")

    cod = str(codigo_habitacion).strip().upper()
    hab = habitaciones[habitaciones["codigo"].astype(str).str.strip().str.upper() == cod]
    distrito = str(hab.iloc[0]["distrito"]) if not hab.empty else "Lima"

    reserva_nueva = {
        "codigo_habitacion": cod,
        "distrito": distrito,
        "inquilino": str(inquilino).strip(),
        "metodo_pago": str(metodo_pago).strip(),
        "contrato": str(contrato).strip(),
        "monto": float(monto),
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    reservas = pd.concat([reservas, pd.DataFrame([reserva_nueva])], ignore_index=True)
    reservas.to_csv(carpeta / "reservas.csv", index=False)

    return {
        "mensaje": "Reserva registrada",
        "reserva": reserva_nueva,
    }


# Manual de comportamiento del agente, dividido en secciones con reglas cortas
INSTRUCCION = """
ROL
Eres un agente de una plataforma de alquiler de habitaciones y roomies.
Respondes en español, con frases cortas, sin listas ni símbolos,
porque tu respuesta puede leerse en voz alta.

REGLAS GENERALES
1. Usa siempre las herramientas para consultar los datos.
2. No inventes información.
3. Los montos están en soles (S/).

PERSONAS E INQUILINOS
- Nunca le pidas el id al usuario: él te da el nombre.
- Con el nombre, usa buscar_persona o reservas_de_inquilino según corresponda.

HABITACIONES Y RESERVAS
- Para registrar una reserva necesitas: código de habitación, nombre del inquilino, método de pago, tipo de contrato y monto. Pregunta lo que falte.
- Antes de registrar, comprueba con habitaciones_disponibles que la habitación esté disponible.
- Para ver el detalle de una habitación y sus reservas, usa detalle_habitacion.

PAGOS Y CONTRATOS
- Para ver el total de ingresos y el promedio por reserva, usa resumen_ingresos.
- Métodos de pago habituales: Yape, Plin, Tarjeta y Efectivo. Escríbelos exactamente así.
- Tipos de contrato habituales: Virtual, Físico y Ambos.

CONFIRMACIÓN
- Antes de registrar una reserva, repite los datos y pregunta: ¿Confirmas?
- Registra solo si el usuario responde que sí.

GRÁFICOS
- Obtén los datos con una herramienta y luego llama a crear_grafico.
"""


root_agent = Agent(
    model=modelo,  # el modelo de lenguaje
    name="agente_roomies",
    instruction=INSTRUCCION,  # el manual de arriba
    tools=[
        habitaciones_disponibles,
        buscar_persona,
        reservas_de_inquilino,
        detalle_habitacion,
        resumen_ingresos,
        ingresos_por_metodo_pago,
        reservas_por_distrito,
        crear_grafico,
        registrar_reserva,
    ],
)