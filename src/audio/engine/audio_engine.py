from src.audio.engine.states import AudioState
from src.core.events import Event, EventType
import time

class AudioEngine:
    #Inicializo el Engine con las variables que pedimos, convertimos los ms a segundos e inicializamos los states y sincronizadores
    def __init__(self, mic, energy, vad, wake_word, recorder,
        frame_ms=32,
        hop_ms=20,):
        self.mic = mic
        self.energy = energy
        self.vad = vad
        self.wake_word = wake_word
        self.recorder = recorder

        self.frame_s = frame_ms / 1000
        self.hop_s = hop_ms / 1000

        self.state = AudioState.INIT
        self.state_ts = time.monotonic()
        self.data = ""
        self.next_tick = time.monotonic()
        self.t_counter = 0
        self.required_hits = int(0.1 / self.hop_s)
        
    #Arranca el engine,  siempre comenzamos calibrando 
    def start(self):
        self.mic.start()
        self.state = AudioState.CALIBRATING     #Se calibra para llenar un buffer en el microfono, creo son 5s
        self.state_ts = time.monotonic()
        print("INICIO")                         #FLAG: Solo para visualizar cuando inicia el modelo
        #Este codigo es el loop principal, libera al cpu del procesamiento innecesario
        while True:
            now = time.monotonic()


            limit = 0
            while now >= self.next_tick and limit < 3:#ejecuta los ticks atrasados 
                
                self._tick()
                
                self.next_tick += self.hop_s
                limit += 1
                
            if now > self.next_tick + (self.hop_s * 5):
                self.next_tick = now

            time.sleep(0.001)
            
        
    #Lo que se ejecuta cada tick
    def _tick(self):
        audio = self.mic.get_last_seconds(self.frame_s) #se toman los ultimos ms que capturo el microfono
        self.wake_word.refresh(self.mic.get_last_seconds(self.frame_s+self.hop_s*(self.wake_word.frames-1)))
        #self.wake_word.push(audio) #apesar de no estar prediciendo, se debe de mandar al modelo del wake word para pre procesar las frecuencias
        #cada estado se llama independiente, no se ejecutan 2 estados en un solo tick, libera a la cpu de carga innecesaria
        if self.state == AudioState.CALIBRATING:
            self._calibrating(audio)

        elif self.state == AudioState.IDLE:
            self._idle(audio)

        elif self.state == AudioState.LISTENING:
            self._listening(audio)

        elif self.state == AudioState.TRIGGERED:
            self._triggered(audio)

        elif self.state == AudioState.RECORDING:
            self._recording(audio)


    def _calibrating(self, audio):
        #Inicializa con al menos 1.5s de audio y se elimina el ruido de los modelos que lo necesiten
        self.stats("Calibrating")
        self.energy.initialize(audio)

        if time.monotonic() - self.state_ts > 1.5:
            self._transition(AudioState.IDLE)       #Pasamos al estado IDLE

    def _idle(self, audio):
        #Este es el estado de espera a cualquier accion, no recibe nada, solo esta al pendiente de algun ruido
        self.stats("IDLE")
        if self.energy.is_voice(audio): #Si escucha algun ruido
            self._transition(AudioState.LISTENING)  #Pasa a la escucha (Listening)
    
    def _listening(self, audio):
        #Este es el estado mas importante, pues debe de decidir si lo que escucha en realidad es voz humana y ademas de reconocer si se dijo la palabra de activacion
        self.stats("Listening:..")
        if not self.energy.is_voice(audio) and self.t_counter == 0:
            self._transition(AudioState.IDLE)   #si el ruido se fue, se regresa al estado IDLE
            self.vad.reset()                    #Se reinicia el vad
            return
        
        if self.vad.is_speech(audio): #Si del frame que pasamos se detecta que es voz humana
            
            score = self.wake_word.predict()        #Se predice con el modelo artesanal si se dijo la palabra clave
            
            self.stats("Listening...",score)
            if score > self.wake_word.threshold:    #Se evalua la prediccion con un umbral y se inicia el contador
                self.t_counter += 1
            else:
                self.t_counter = 0
            
            if self.t_counter > self.required_hits:
                self.t_counter = 0
                self.stats("Listening...", "KORA!") 
                self._transition(AudioState.TRIGGERED)#se pasa a captado (Triggered)



    def _triggered(self, audio):
        self.stats("Triggered")
        
        #Se emite un bus con el evento WAKE_WORD que debe de funcionar como notificador de que la palabra fue dicha por el usuario
        """self.bus.emit(
            Event(EventType.WAKE_WORD, {"timestamp": time.time()})
        )"""
        self.recorder.start()                               #se comienza a grabar el comando del usuario
        self.wake_word.save_window("Data/audio/captured")
        self._transition(AudioState.RECORDING)              #se pasa a grabar (Recording)

    def _recording(self, audio):
        self.stats("Recording")
        self.recorder.append(audio)             #el fragmento de audio se acumula

        if self.recorder.should_stop():         #Si se detecta que se dejo de hablar
            clip = self.recorder.stop()         #se para la grabacion y se toma el clip grabado

            #Se emite un bus con el comando dicho por el usuario
            """self.bus.emit(
                Event(
                    EventType.COMMAND_AUDIO,
                    {
                        "audio": clip,
                        "sample_rate": self.mic.sample_rate,
                    },
                )
            )"""
            

            self.stats("Recording", "grabado")
            self._transition(AudioState.IDLE)   #se regresa a IDLE

    def _transition(self, new_state):
        #Se registra el cambio y se guarda el momento en que se realizo
        self.state = new_state
        self.state_ts = time.monotonic()

        
    

    def stats(self, mode, data = ""):
        #solo muestra la interaccion
        if data != "":
            self.data = "| "+str(data)
        print(f"\r{mode} {self.data}                                  ", end="", flush=True)