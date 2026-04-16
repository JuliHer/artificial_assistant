import yaml
from pathlib import Path

from src.processing.DataLoader import Loader
from src.processing.DataProcessing import Process 
from src.training.IntentionTraining import IntentTraining 
from src.training.NERTraining import NERTraining
from src.training.wake_word_training import WakeWordTraining
from src.inference.KORA import KORA

import src.core.skill_loader as loader
import src.core.skill_resolver as resolver
import src.core.policy as policy
import src.core.dispatcher as dispatcher
import src.core.normalizer as normalizer
from src.core.pending_handler import PendingHandler

from src.core.session import SessionContext
from src.core.context import ActionContext
from src.core.action_result import ActionStatus, ActionResult
import matplotlib.pyplot as plt
import numpy as np

class Main:
    def __init__(self):

        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        with open(str(project_root / "src" / "configs" / "config.yaml"), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.training_intent = IntentTraining()
        self.training_NER = NERTraining()
        self.training_WWD = WakeWordTraining()
        self.dataLoader = Loader()
        self.brain = KORA()
        self.menu()
        pass


    def menu(self):
        op = 1
        while op != 0:
            print("Este es el menu principal para KORA")
            print("Selecciona una opccion.")
            print("1. Entrenar modelo")
            print("2. Probar modelos")
            print("0. Salir")
            op = int(input("TU: "))

            if op == 1:
                
                print("Que modelo quieres entrenar?")
                print("1. KORA Intentions Classifier")
                print("2. KORA Named Entity Recognition")
                print("3. KORA Wake Word Detector")
                op = int(input("TU: "))
                if op == 1:
                    X, y = self.dataLoader.get_data_intentions()

                    model = self.training_intent.create_neuronal(sorted(list(set(y))))
                    self.training_intent.fit(model, X, y)
                elif op == 2:
                    X, y = self.dataLoader.get_data_ner()
                    self.training_NER.create_neuronal()
                    self.training_NER.fit(X,y)
                elif op == 3:
                    X_train, X_test, Y_train, Y_test = self.dataLoader.get_data_wake_word(42)
                    self.training_WWD.create_neuronal()
                    
                    
                    self.training_WWD.fit(X_train, X_test, Y_train, Y_test, plot=True)

                self.brain.load()
            elif op == 2:
                print("Modelo de reconocimiento de intenciones y de entidades nombradas")
                print("para salir escribe 'exit'")
                print("KORA: Escribe una instruccion para el vehiculo")
                entrada = ''
                pending_handler = PendingHandler()
                session_ctx = SessionContext()
                while entrada != 'exit':    
                    entrada = input("TU: ")
                    context = self.brain.predict(entrada)
                    context.debug.log("PREDICT", {
                        "intent": context.intent,
                        "confidece": context.intent_confidence,
                        "entities":context.entities
                    })
                    context.session = session_ctx.session 
                    context.raw_text = entrada

                    if Main.is_cancel(context) and pending_handler.is_pending(context):
                        pending_handler.clear(context)
                        self.speak("Acción cancelada")
                        
                        continue
                        

                    if pending_handler.is_pending(context):
                        if pending_handler.is_timed_out(context):
                            pending_handler.clear(context)
                            self.speak("Se agotó el tiempo, cancelé la acción")
                            continue

                    skills = loader.load_skills()
                    skill = resolver.resolve_skill(context, skills)

                    context.debug.log("SKILL_RESOLVER", {
                        "candidates": [s.name for s in skills],
                        "selected": skill.name if skill else None
                    })

                    if pending_handler.is_pending(context):
                        pending_action = context.session["pending_action"]
                        skill = pending_action["skill"]
                        if not skill:
                            self.speak("no se encontro skill compatible")
                            pending_handler.clear(context)
                            continue
                        
                        context.session["translator"] = normalizer.normalize(context, skill, self.brain.model)

                        result = dispatcher.resume(context, skill, pending_action)
                        
                        if result.status == ActionStatus.SUCCESS:
                            self.speak(result.message)
                        elif result.status == ActionStatus.FAILED:
                            self.speak(result.message)
                            
                        elif result.status in [ActionStatus.CONFIRM_REQUIRED, ActionStatus.NEEDS_INPUT]:

                            pending_handler.create_pending(context, skill, result.status, result.message, result.data)
                            self.speak(result.message)
                            if self.config["debug"]["enabled"]:
                                context.debug.dump()
                            continue
                                
                        
                        pending_handler.clear(context)
                        if self.config["debug"]["enabled"]:
                            context.debug.dump()
                        continue

                    if not skill:
                        self.speak("no se encontro skill compatible")
                        if self.config["debug"]["enabled"]:
                            context.debug.dump()
                        continue

                    context.session["translator"] = normalizer.normalize(context, skill, self.brain.model)
                    
                    policyresult = policy.evaluate(context, skill)
                    context.debug.log("POLICY", {
                        "decision": policyresult.decision,
                        "reason": policyresult.message
                    })

                    if policyresult.decision == "ASK":
                        pending_handler.create_pending(context, skill, "ASK", policyresult.message, missing=policyresult.missing)
                        self.speak(policyresult.message)
                        if self.config["debug"]["enabled"]:
                            context.debug.dump()
                        continue
                        
                    if policyresult.decision == "DENY":
                        self.speak(policyresult.message)
                        if self.config["debug"]["enabled"]:
                            context.debug.dump()
                        continue
                    context.debug.log("DISPATCH", {
                        "skill": skill.name,
                        "type": skill.type
                    })

                    result = dispatcher.dispatch(context, skill)

                    if result.status == ActionStatus.SUCCESS:
                        self.speak(result.message)
                    elif result.status == ActionStatus.FAILED:
                        self.speak(result.message)
                    elif result.status in [ActionStatus.CONFIRM_REQUIRED, ActionStatus.NEEDS_INPUT]:
                        pending_handler.create_pending(context, skill, result.status, result.message, result.data)
                        self.speak(result.message)
                        if self.config["debug"]["enabled"]:
                            context.debug.dump()
                        continue
                    
                    pending_handler.clear(context)
                    if self.config["debug"]["enabled"]:
                        context.debug.dump()
                    
    def speak(self,text):
        print(f"KORA: {text}")
    



    def is_cancel(context):
        return context.intent == "cancel"








Main()
