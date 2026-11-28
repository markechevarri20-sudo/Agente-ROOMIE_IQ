from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google.adk.runners import InMemoryRunner

from agente import root_agent

app = FastAPI()
ejecutor = InMemoryRunner(agent=root_agent)   # el que le pasa las preguntas al agente


# Forma de la pregunta que envía la página: {"texto": "..."}
class Pregunta(BaseModel):
    texto: str


# GET /  ->  muestra la página del chat
@app.get("/")
def mostrar_pagina():
    # no-store: el navegador no guarda copia de la página, siempre carga la última
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store"})


# POST /chat  ->  recibe la pregunta, se la entrega al agente y devuelve su respuesta
@app.post("/chat")
async def chat(pregunta: Pregunta):
    eventos = await ejecutor.run_debug(pregunta.texto, quiet=True)

    texto_respuesta = ""
    datos_grafico = None
    for evento in eventos:
        # ¿El agente usó la herramienta crear_grafico?
        for resultado in evento.get_function_responses():
            if resultado.name == "crear_grafico":
                datos_grafico = resultado.response
        # El evento final trae el texto de la respuesta
        if evento.is_final_response():
            texto_respuesta = evento.content.parts[0].text

    return {"respuesta": texto_respuesta, "grafico": datos_grafico}
