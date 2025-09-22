import os
import re
import sys
import json
import torch
from unsloth import FastLanguageModel
import datetime
import unicodedata

# -----------------------
# Configuración de rutas
# -----------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
# BASE_MODEL_DIR = os.path.join(ROOT, "mistral-7b-bnb-4bit") Si tienes el modelo en una carpeta
#Si tienes el modelo en la raiz del proyecto
BASE_MODEL_DIR = os.path.abspath(os.path.join(ROOT, "..", "mistral-7b-bnb-4bit"))
LORA_ADAPTER_DIR = os.path.join(ROOT, "mistral_finetuned_miramar_combined_steps20000")
COTIZACIONES_FILE = os.path.join(ROOT, "cotizaciones.json")

#Extras#
_LUGAR_WORD = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\.\-'\s]+"

LOCATION_EXCLUDE = {
    "sin regreso",
    "solo ida",
    "ida",
    "ida y vuelta",
    "ida y regreso",
    "no",
    "ninguno",
    "ninguna",
    "sin vuelta",
    "sin retorno",
    "no regreso",
}

SPANISH_NUMBER_WORDS = {
    "un": 1,
    "una": 1,
    "uno": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
}

_NUM_WORD_PATTERN = "|".join(sorted(SPANISH_NUMBER_WORDS.keys(), key=len, reverse=True))

WORD_CONTEXT_PATTERNS = [
    re.compile(
        rf"(?:somos|vamos|seremos|seran|viajamos|viajan|van|iremos|iran)\s+({_NUM_WORD_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b({_NUM_WORD_PATTERN})\b\s*(?:personas?|pasajeros?|viajeros?|clientes|ocupantes)",
        re.IGNORECASE,
    ),
]

WORD_DIGIT_PATTERN = re.compile(
    r"(?:somos|vamos|seremos|seran|serán|viajamos|viajan|van|iremos|iran|irán)\s+(\d{1,3})\b",
    re.IGNORECASE,
)

QUANTITY_STANDALONE_DIGIT_RE = re.compile(r"^\s*(\d{1,3})\s*$")


def _question_mentions_forbidden(question: str, missing: list[str]) -> bool:
    q = question.lower()
    forbidden = []
    if "origen" not in missing:
        forbidden.extend([
            "origen",
            "desde",
            "punto de partida",
            "de dónde",
            "donde parte",
            "inicio del viaje",
        ])
    if "destino" not in missing:
        forbidden.extend([
            "destino",
            "hasta",
            "hacia",
            "a dónde",
            "donde llega",
        ])
    if "regreso" not in missing:
        forbidden.extend([
            "regreso",
            "vuelta",
            "retorno",
        ])
    if "fecha" not in missing:
        forbidden.extend([
            "fecha",
            "cuándo viajas",
        ])
    if "hora" not in missing:
        forbidden.extend([
            "hora",
            "a qué hora",
        ])
    if "cantidad" not in missing:
        forbidden.extend([
            "cuántas personas",
            "personas viajarán",
            "pasajeros",
        ])
    for token in forbidden:
        if token in q:
            return True
    return False


def _select_question_fields(missing: list[str]) -> list[str]:
    ordered = ["origen", "destino", "fecha", "hora", "regreso", "cantidad"]
    missing_set = set(missing)
    if not missing_set:
        return []
    if {"origen", "destino"}.issubset(missing_set):
        return ["origen", "destino"]
    if {"fecha", "hora"}.issubset(missing_set):
        return ["fecha", "hora"]
    if {"regreso", "cantidad"}.issubset(missing_set):
        return ["regreso", "cantidad"]
    selected: list[str] = []
    for field in ordered:
        if field in missing_set:
            selected.append(field)
            if len(selected) == 2:
                break
    return selected


def _build_fallback_question(missing: list[str]) -> str | None:
    ordered = ["origen", "destino", "fecha", "hora", "regreso", "cantidad"]
    remaining = [item for item in ordered if item in missing]
    if not remaining:
        return None

    parts: list[str] = []

    if "origen" in remaining and "destino" in remaining:
        parts.append("desde qué lugar parte tu viaje y cuál es el destino")
        remaining = [item for item in remaining if item not in {"origen", "destino"}]
    elif "origen" in remaining:
        parts.append("desde qué lugar parte tu viaje")
        remaining.remove("origen")
    elif "destino" in remaining:
        parts.append("hacia dónde necesitas viajar")
        remaining.remove("destino")

    if len(parts) < 2 and "fecha" in remaining and "hora" in remaining:
        parts.append("para qué fecha y a qué hora será el viaje")
        remaining = [item for item in remaining if item not in {"fecha", "hora"}]

    if len(parts) < 2 and "fecha" in remaining:
        parts.append("en qué fecha tienen previsto el viaje")
        remaining.remove("fecha")

    if len(parts) < 2 and "hora" in remaining:
        parts.append("a qué hora desean iniciar el viaje")
        remaining.remove("hora")

    if len(parts) < 2 and "regreso" in remaining and "cantidad" in remaining:
        parts.append("si es solo ida o incluye regreso y cuántas personas viajarán")
        remaining = [item for item in remaining if item not in {"regreso", "cantidad"}]

    if len(parts) < 2 and "regreso" in remaining:
        parts.append("si el servicio es solo ida o incluye regreso")
        remaining.remove("regreso")

    if len(parts) < 2 and "cantidad" in remaining:
        parts.append("cuántas personas viajarán")
        remaining.remove("cantidad")

    if not parts:
        return None
    body = parts[0] if len(parts) == 1 else f"{parts[0]} y {parts[1]}"
    return f"¿{body}?"


_PLACE_PREFIX_RE = re.compile(r"^(?:origen|destino|desde|hacia|hasta|rumbo a|rumbo|direcci[óo]n|punto de partida|punto de destino)\s*[:\-]?\s*",
                               re.IGNORECASE)

MONTHS = {
    # español
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12,
    # abreviaturas comunes
    "ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,"jul":7,"ago":8,"sep":9,"set":9,"oct":10,"nov":11,"dic":12,
    # inglés básico por si acaso
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"abr":4,"aug":8,"sept":9,"oct":10,"nov":11,"dec":12,
}
WEEKDAYS = {
    "lunes":0,"martes":1,"miércoles":2,"miercoles":2,"jueves":3,"viernes":4,"sábado":5,"sabado":5,"domingo":6,
    # inglés por si aparece
    "monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6,
}

MONTH_PATTERN = "|".join(
    sorted({re.escape(k) for k in MONTHS.keys()}, key=len, reverse=True)
)

DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b\d{{1,2}}\s+de\s+({MONTH_PATTERN})\b", re.IGNORECASE),
    re.compile(rf"\b({MONTH_PATTERN})\s+\d{{1,2}}\b", re.IGNORECASE),
]

DATE_KEYWORDS = {
    "hoy",
    "mañana",
    "pasado mañana",
    "pasado-manana",
    "proximo",
    "próximo",
    "este",
}

TIME_PATTERNS = [
    re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm|hrs|horas|h)?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s*(?:am|pm)\b", re.IGNORECASE),
    re.compile(r"\b(?:medianoche|mediod[ií]a|mediodia)\b", re.IGNORECASE),
]

REGRESO_NO_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsin\s+regreso\b",
        r"\bsin\s+vuelta\b",
        r"\bsin\s+retorno\b",
        r"\bsolo\s+ida\b",
        r"\bida\s+solamente\b",
        r"\bno\s+regreso\b",
    ]
]

REGRESO_YES_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcon\s+regreso\b",
        r"\bida\s+y\s+vuelta\b",
        r"\bida\s+y\s+regreso\b",
        r"\bcon\s+retorno\b",
        r"\bcon\s+vuelta\b",
    ]
]

def _titlecase_clean(s: str) -> str:
    s = s.strip().strip(".,;:!¿?'()[]{}")
    # Capitaliza palabras >2 letras
    return " ".join(w.capitalize() if len(w) > 2 else w.lower() for w in s.split())

def _clean_location_noise(val: str) -> str:
    # Quita frases comunes que ensucian el valor
    val = re.sub(r"\b(de\s+mi\s+viaje|mi\s+viaje|del?\s+viaje)\b", "", val, flags=re.IGNORECASE).strip()
    # Corta por conectores/comas si sobran frases
    _split_pat = re.compile(r"(,| que | quiero | es | sera | será | con | hasta )", re.IGNORECASE)
    val = _split_pat.split(val)[0].strip()
    # Limpieza final y capitalización
    val = _titlecase_clean(val)
    return val

def _sanitize_print(s: str) -> str:
    s = s.replace("\r", " ").replace("\x00", "").strip()
    # Normaliza espacios para evitar caracteres invisibles
    return unicodedata.normalize("NFKC", s)


def _strip_accents(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))


def normalize_place(value: str) -> str:
    if not value:
        return ""
    txt = unicodedata.normalize("NFKC", value).strip()
    txt = _PLACE_PREFIX_RE.sub("", txt)
    cleaned = _clean_location_noise(txt)
    return cleaned


def normalize_regreso(value: str) -> str | None:
    if not value:
        return None
    txt = unicodedata.normalize("NFKC", value).strip().lower()
    txt = re.sub(r"[^a-záéíóúñü\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return None
    no_phrases = [
        "sin regreso",
        "sin vuelta",
        "sin retorno",
        "solo ida",
        "ida solo",
        "ida solamente",
        "no regreso",
        "no llevamos regreso",
        "no tenemos regreso",
    ]
    yes_phrases = [
        "con regreso",
        "ida y regreso",
        "ida y vuelta",
        "con retorno",
        "con vuelta",
        "con regreso incluido",
    ]
    if txt in {"no", "n", "no gracias"} or any(phrase in txt for phrase in no_phrases):
        return "no"
    if txt in {"si", "sí", "s"} or any(phrase in txt for phrase in yes_phrases):
        return "sí"
    return None


_QUANTITY_DIGITS_CONTEXT_RE = re.compile(
    r"\b(\d{1,3})\b\s*(?:personas?|pasajeros?|viajeros?|clientes|ocupantes)",
    re.IGNORECASE,
)


def extract_quantity(user_input: str) -> int | None:
    if not user_input:
        return None
    normalized = unicodedata.normalize("NFKC", user_input)
    simple = _strip_accents(normalized).lower()

    # Busca dígitos acompañados de contexto
    m = _QUANTITY_DIGITS_CONTEXT_RE.search(normalized)
    if m:
        try:
            val = int(m.group(1))
            if val > 0:
                return val
        except ValueError:
            pass

    m = WORD_DIGIT_PATTERN.search(normalized)
    if m:
        try:
            val = int(m.group(1))
            if val > 0:
                return val
        except ValueError:
            pass

    m = QUANTITY_STANDALONE_DIGIT_RE.match(normalized)
    if m:
        try:
            val = int(m.group(1))
            if val > 0:
                return val
        except ValueError:
            pass

    for pattern in WORD_CONTEXT_PATTERNS:
        m = pattern.search(simple)
        if m:
            word = m.group(1)
            number = SPANISH_NUMBER_WORDS.get(word)
            if number:
                return number
    return None


def normalize_fecha(value: str, context: str) -> str:
    if not value:
        return ""
    text = value.strip()
    try:
        dt = datetime.datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return value
    if re.search(r"\b20\d{2}\b", context):
        return text
    today = datetime.date.today()
    date_only = dt.date()
    if date_only >= today:
        return text
    candidate = datetime.date(today.year, date_only.month, date_only.day)
    if candidate < today:
        candidate = datetime.date(today.year + 1, date_only.month, date_only.day)
    return candidate.isoformat()


def user_mentions_date(user_input: str) -> bool:
    if not user_input:
        return False
    text = unicodedata.normalize("NFKC", user_input)
    lowered = text.lower()
    for pat in DATE_PATTERNS:
        if pat.search(text):
            return True
    for word in DATE_KEYWORDS:
        if word in lowered:
            return True
    return False


def user_mentions_time(user_input: str) -> bool:
    if not user_input:
        return False
    text = unicodedata.normalize("NFKC", user_input)
    for pat in TIME_PATTERNS:
        if pat.search(text):
            return True
    return False


def user_mentions_regreso(user_input: str) -> bool:
    if not user_input:
        return False
    text = unicodedata.normalize("NFKC", user_input)
    for pat in REGRESO_NO_PATTERNS + REGRESO_YES_PATTERNS:
        if pat.search(text):
            return True
    return False

def _pick_last_capitalized_chunk(text: str) -> str | None:
    # Como último recurso: toma la última “palabra o 2-3 palabras” con pinta de nombre propio
    tokens = [t for t in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-']+", text)]
    if not tokens:
        return None
    # toma las últimas 1-3
    last = tokens[-3:]
    candidate = " ".join(last).strip()
    return _titlecase_clean(candidate) if candidate else None

def extract_location(user_input: str, field: str = "origen") -> str | None:
    txt = user_input.strip()
    if field == "origen":
        # Solo origen
        m = re.search(r"(desde|parto desde|salgo de|partida de|salida de|en)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s']+)", txt, re.IGNORECASE)
        if m:
            val = normalize_place(m.group(2))
            if val and len(val) >= 2 and val.lower() not in LOCATION_EXCLUDE:
                return val
        # Si solo escribió el lugar
        m = re.search(r"^([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s']+)$", txt)
        if m:
            val = normalize_place(m.group(1))
            if val and len(val) >= 2 and val.lower() not in LOCATION_EXCLUDE:
                return val
    elif field == "destino":
        # Solo destino
        m = re.search(r"(hacia|a|destino\s+(?:es|ser[áa]))\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-\s']+)", txt, re.IGNORECASE)
        if m:
            val = normalize_place(m.group(2))
            if val and len(val) >= 2 and val.lower() not in LOCATION_EXCLUDE:
                return val
    return None


_ORIG_DEST_PATTERNS = [
    re.compile(
        rf"(?:desde|parto desde|salgo de|partida de|salida de)\s+(?P<orig>{_LUGAR_WORD})(?:\s+y\s+(?:vamos|voy|iremos|iré|queremos ir|quiero ir|queremos viajar|quiero viajar|viajamos|viajaré)\s+a\s+(?P<dest_y>{_LUGAR_WORD})|\s+hasta\s+(?P<dest_hasta>{_LUGAR_WORD})|\s+a\s+(?P<dest_a>{_LUGAR_WORD})|\s+hacia\s+(?P<dest_hacia>{_LUGAR_WORD}))",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:vamos|voy|iremos|iré|queremos ir|quiero ir|queremos viajar|quiero viajar|viajamos)\s+desde\s+(?P<orig>{_LUGAR_WORD})\s+(?:a|hacia|hasta)\s+(?P<dest_generic>{_LUGAR_WORD})",
        re.IGNORECASE,
    ),
]


def extract_origin_dest_pair(user_input: str) -> tuple[str | None, str | None]:
    txt = user_input.strip()
    for pattern in _ORIG_DEST_PATTERNS:
        m = pattern.search(txt)
        if not m:
            continue
        orig_raw = m.groupdict().get("orig")
        dest_raw = None
        for key in ["dest_y", "dest_hasta", "dest_a", "dest_hacia", "dest_generic"]:
            val = m.groupdict().get(key)
            if val:
                dest_raw = val
                break
        if not orig_raw or not dest_raw:
            continue
        orig = normalize_place(orig_raw)
        dest = normalize_place(dest_raw)
        if (
            orig
            and dest
            and orig.lower() not in LOCATION_EXCLUDE
            and dest.lower() not in LOCATION_EXCLUDE
        ):
            return orig, dest
    return None, None

# -----------------------
# Guardado de cotizaciones
# -----------------------
def guardar_cotizacion(state: dict):
    data = []
    if os.path.exists(COTIZACIONES_FILE):
        try:
            with open(COTIZACIONES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    data.append(state)
    with open(COTIZACIONES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\nTransportes Miramar: 📁 Cotización guardada en {COTIZACIONES_FILE}")

# -----------------------
# Carga de modelo
# -----------------------
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit = device == "cuda"
    print("Cargando modelo base y adaptador LoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_DIR,
        adapter_name=LORA_ADAPTER_DIR,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=use_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Modelo cargado.\n")
    return model, tokenizer, device

BASE_STYLE = """
Rol: Eres un chatbot de Transportes Miramar, especializado en guiar al cliente en el proceso de cotización de viajes, vendedor cordial y profesional de Transportes Miramar (Chile).
Canal: conversación tipo WhatsApp con mensajes breves, amables y claros.
Objetivo: guiar al cliente paso a paso para cotizar un viaje.
Tono: cercano, profesional, directo; evita tecnicismos. No uses emojis salvo que el usuario los use antes.
Prohibido: precios, datos personales, políticas internas, información no pedida o inventada.
Formato: responde SOLO con el contenido solicitado entre las etiquetas pedidas. Nada fuera de esas etiquetas.
Importante: el estilo debe ser MUY parecido a las referencias, pero NO copies literalmente ningún ejemplo.
Varía levemente la redacción en cada turno para sonar natural y no repetitivo.
Regla especial: para origen y destino, acepta cualquier ubicación que el cliente indique (ciudad, calle, referencia, etc.) como válida. Nunca pidas dirección exacta ni más precisión, y nunca insistas si ya se indicó una ubicación. Nunca preguntes por la ubicación actual del usuario ni por "dirección específica" ni por "dirección exacta". Si el usuario indica cualquier lugar, acéptalo tal cual, sin pedir detalles adicionales.
Nunca preguntes por la ubicación actual del usuario ni por dirección exacta ni por dirección específica. Si el usuario indica cualquier lugar, acéptalo tal cual, sin pedir detalles adicionales.
Nunca generes frases como "¿En qué ciudad o dirección específica te encuentras en este momento?", "¿Dónde estás ahora?", "¿Me puedes dar la dirección exacta?", "¿Cuál es tu ubicación actual?", ni variantes similares. Si el usuario indica cualquier lugar, acéptalo tal cual, sin pedir detalles adicionales.
Alcance: SOLO puedes hablar de Transportes Miramar y del proceso de cotizar viajes con la empresa. Si el usuario pide algo distinto, responde brevemente que tu misión es gestionar cotizaciones de Transportes Miramar y redirígelo a ese objetivo.
"""

# -----------------------
# Bot conversacional
# -----------------------
class MiramarSellerBot:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.reset()

    def reset(self):
        self.state = {
            "origen": "",
            "destino": "",
            "fecha": "",
            "hora": "",
            "regreso": "",
            "cantidad": 0,
            "especial": "",
        }
        self.step = 0
        self.saved = False

    @staticmethod
    def decode_new_only(tokenizer, outputs, inputs):
        seq = outputs[0]
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = seq[prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def next_missing(self):
        keys = ["origen", "destino", "fecha", "hora", "regreso", "cantidad"]
        for key in keys:
            if key == "cantidad" and self.state[key] == 0:
                return key
            if key != "cantidad" and not self.state[key]:
                return key
        return None

    def validate_input_llm(self, user_input):
        examples = """
    Ejemplos:
    - Entrada: "Vamos el 10/10 a las 2pm" → {{"fecha": "2025-10-10", "hora": "14:00"}}
    - Entrada: "Vamos el 10/10 a las 2pm , sin ningun comentario adicional" → {{"fecha": "2025-10-10", "hora": "14:00", "especial": "no"}}
    - Entrada: "Sin regreso, vamos 3 personas" → {{"regreso": "no", "cantidad": 3}}
    - Entrada: "Solo ida, somos 2" → {{"regreso": "no", "cantidad": 2}}
    - Entrada: "Sin requerimiento especial" → {{"especial": "no"}}
    """
        prompt = (
            BASE_STYLE
            + f"""
    Tarea: EXTRAER y NORMALIZAR datos de cotización en Chile desde la respuesta del cliente.
    Devuelve un JSON con SOLO las claves presentes y válidas de:
    - origen (str), destino (str), fecha (YYYY-MM-DD), hora (HH:MM 24h), regreso ('sí'/'no' o 'ida'/'ida y regreso'), cantidad (int), especial (str).
    Reglas:
    - Para regreso, acepta y normaliza variantes como "sin regreso", "solo ida", "no regreso", "es solo ida", "ida", "ida y vuelta", "ida y regreso", "regreso", "no", "sí", etc. Si el usuario indica que no hay regreso, normaliza como "no" o "solo ida".
    - Si el usuario responde con una negativa, evasiva o fuera de contexto (por ejemplo: "no", "no quiero", "prefiero no responder", "no sé", "no por ahora"), NO extraigas ningún dato y devuelve {{}}.
    - No inventes ni completes con supuestos.
    - Acepta fechas tipo “27 de junio”, “27/06/2025”, “27-06”, “mañana”, “el 10/10 a las 2pm”, “salimos el 10/10 a las 2pm”, “vamos el 10 de octubre a las 14:00”, etc. (normaliza a YYYY-MM-DD y HH:MM 24h).
    - Si el usuario escribe fecha y hora juntas, extrae ambos campos correctamente.
    - Si el usuario responde con varios datos juntos (ejemplo: "Sin regreso, vamos 3 personas" o "Vamos el 10/10 a las 2pm , sin ningún comentario adicional"), extrae todos los campos presentes.
    - Ignora frases como "sin ningún comentario adicional", "sin requerimiento especial", "ningún comentario", "ningún requerimiento" para el campo especial (normaliza como "no").
    - Para cantidad, acepta variantes como "vamos 3 personas", "somos 3", "viajamos 3", "seremos 3", etc.
    - Para origen y destino, acepta cualquier ubicación que el cliente indique (ciudad, calle, referencia, etc.) como válida. Nunca pidas dirección exacta ni más precisión.
    - Si no hay datos válidos, devuelve {{}}.
    {examples}
    Contexto (estado actual): {json.dumps(self.state, ensure_ascii=False)}
    Respuesta del cliente: {user_input}
    JSON:
    """
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = MiramarSellerBot.decode_new_only(self.tokenizer, outputs, inputs)
        m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}


    def generate_greeting_llm(self) -> str:
        prompt = (BASE_STYLE + """
Tarea: Genera el primer mensaje para saludar cordialmente al usuario.
Objetivo del mensaje: dar la bienvenida y ofrecer ayuda sin pedir datos aún.
Guía de estilo:
- El mensaje debe ser cálido, profesional, claro y bien redactado.
- No repitas ni copies frases de ejemplos anteriores.
- No uses ninguna estructura ni frase que hayas visto antes.
- Genera una variante original, diferente en cada turno, pero manteniendo el mismo objetivo y tono.
Reglas:
- La redacción debe ser impecable, sin errores ortográficos ni gramaticales.
- No generes frases cortadas, ambiguas ni poco claras.
- Revisa mentalmente la frase antes de responder, como lo haría un profesional.
- Si tienes dudas, prioriza la claridad y corrección.
- Una sola oración. Termina en pregunta.
- Empieza con un saludo cordial (por ejemplo, “Hola” o “Hola, buen día”) y ofrece tu ayuda.
- No pidas información de viaje todavía (no menciones origen, destino, fecha, hora, regreso ni cantidad en este primer mensaje).
- Mantén el tono cálido, profesional y amistoso.
- Después de saludar, pregunta si desea cotizar un viaje o cómo podrías ayudarlo.
- Menciona explícitamente que hablas en nombre de Transportes Miramar.
- Mantén un tono cálido y profesional.
- Varía levemente la redacción en cada turno para sonar natural y no repetitivo.
- No uses la misma estructura dos veces seguidas.
- La frase debe ser gramaticalmente correcta y fácil de entender.
Entrega: <msg>…</msg>
<msg>
""")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = MiramarSellerBot.decode_new_only(self.tokenizer, outputs, inputs)
        m = re.search(r"<msg>(.*?)</msg>", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            greeting = re.sub(r"</?msg>", "", m.group(1), flags=re.IGNORECASE).strip()
        else:
            greeting = re.sub(r"</?msg>", "", text, flags=re.IGNORECASE).strip()
        if re.search(
            r"en qué lugar te encuentras|dónde estás|donde te encuentras|desde qué lugar",
            greeting,
            re.IGNORECASE,
        ):
            return self.generate_greeting_llm()
        return greeting

    def initial_prompt(self) -> str:
        greeting = self.generate_greeting_llm()
        if greeting:
            return greeting
        return (
            "Hola, gracias por contactar a Transportes Miramar. Para ayudarte con la cotización te pediré algunos datos."
        )

    def get_next_question(self):
        missing = []
        for k in ["origen", "destino", "fecha", "hora", "regreso", "cantidad"]:
            if k == "cantidad":
                if self.state[k] == 0:
                    missing.append(k)
            elif not self.state[k]:
                missing.append(k)
        if not missing:
            return None
        question_fields = _select_question_fields(missing)
        if not question_fields:
            question_fields = missing[:2]
        confirmed = {
            k: self.state[k]
            for k in ["origen", "destino", "fecha", "hora", "regreso", "cantidad"]
            if (k == "cantidad" and self.state[k] > 0)
            or (k != "cantidad" and self.state[k])
        }
        prompt = (
            BASE_STYLE
            + f"""
Contexto de la cotización (estado): {self.state}
Datos confirmados: {confirmed}
Campos faltantes para esta pregunta: {question_fields}
Tarea: escribe UNA sola pregunta que avance la conversación y pida únicamente esos campos listados.
Reglas clave:
- Expresa la pregunta en tono cordial tipo WhatsApp y en una sola oración.
- No incluyas saludos ni despedidas; comienza directamente con la pregunta.
- Si faltan origen y destino, menciónalos juntos en la misma frase usando formulaciones como “¿Desde qué lugar inicia tu viaje y cuál es el destino?”.
- No uses ni insinúes expresiones como “¿dónde estás?”, “¿en qué lugar te encuentras?”, “dirección exacta” o “ubicación actual”.
- Si un campo ya está completo en el estado, prohíbete mencionarlo o confirmarlo; enfócate solo en los elementos listados en Campos faltantes.
- No reformules ni retomes datos listados en Datos confirmados; asume que ya quedaron claros.
- Si falta “regreso”, pregunta únicamente si el viaje es solo ida o incluye regreso (ej.: “¿El viaje es solo ida o también incluye regreso?”) y no menciones horarios de regreso ni ciudades adicionales.
- Si faltan “regreso” y “cantidad”, puedes unirlos en la misma oración manteniendo primero la pregunta sobre el regreso y luego la cantidad de personas.
- Pide como máximo DOS datos en la misma pregunta; si faltan más, prioriza según el orden del listado en Campos faltantes.
- Puedes agrupar hasta dos campos en la misma pregunta (por ejemplo fecha+hora o regreso+cantidad) si faltan ambos.
- Termina siempre con signo de interrogación de cierre.
- Mantén la redacción fluida, evita copiar literalmente instrucciones de este prompt y no repitas siempre la misma estructura.
Entrega: <q>…</q>

<q>
"""
        )
        pregunta = self._generate_question_with_prompt(prompt, question_fields)
        if pregunta:
            return pregunta
        fallback = _build_fallback_question(question_fields)
        if fallback:
            return fallback
        return "¿Podrías confirmarme el dato que nos falta para seguir con la cotización?"

    def _generate_question_with_prompt(self, prompt: str, question_fields: list[str]) -> str | None:
        for attempt in range(2):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            text = MiramarSellerBot.decode_new_only(self.tokenizer, outputs, inputs)
            m = re.search(r"<q>(.*?)</q>", text, flags=re.IGNORECASE | re.DOTALL)
            pregunta = m.group(1).strip() if m else text.strip()
            pregunta = re.sub(r"</?q>", "", pregunta, flags=re.IGNORECASE).strip()
            if not pregunta:
                return None
            if not _question_mentions_forbidden(pregunta, question_fields):
                return pregunta
            # Refuerza instrucciones para reintentar sin mencionar datos confirmados
            prompt = (
                prompt
                + "\nImportante: acabas de mencionar datos ya confirmados. Reescribe la pregunta mencionando únicamente los campos listados en Campos faltantes.\n<q>"
            )
        return None

    def _generate_confirmation_llm(self, actualizados: dict) -> str | None:
        prompt = (
            BASE_STYLE
            + f"""
Contexto del viaje: {self.state}
Datos nuevos que debes confirmar: {actualizados}
Tarea: redacta UNA sola frase breve confirmando únicamente estos datos nuevos, sin pedir información adicional.
Reglas:
- Menciona solo los campos incluidos en "Datos nuevos".
- Usa un tono cordial tipo WhatsApp.
- No repitas datos ya confirmados previamente ni agregues preguntas.
- No inventes información ni añadas campos extra.
- Entrega el mensaje envuelto en <conf>…</conf>.

<conf>
"""
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = MiramarSellerBot.decode_new_only(self.tokenizer, outputs, inputs)
        m = re.search(r"<conf>(.*?)</conf>", text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        message = re.sub(r"</?conf>", "", m.group(1), flags=re.IGNORECASE).strip()
        return message or None

    def _build_confirmation_fallback(self, actualizados: dict) -> str:
        partes = []
        if "origen" in actualizados and "destino" in actualizados:
            partes.append(
                f"Anoté el recorrido desde {self.state['origen']} hasta {self.state['destino']}."
            )
        else:
            if "origen" in actualizados:
                partes.append(f"Registré el origen como {self.state['origen']}.")
            if "destino" in actualizados:
                partes.append(f"Registré el destino como {self.state['destino']}.")
        if "fecha" in actualizados and "hora" in actualizados:
            partes.append(
                f"La salida queda para el {self.state['fecha']} a las {self.state['hora']}."
            )
        else:
            if "fecha" in actualizados:
                partes.append(f"La fecha queda para el {self.state['fecha']}.")
            if "hora" in actualizados:
                partes.append(f"La hora queda fijada a las {self.state['hora']}.")
        if "regreso" in actualizados:
            if self.state["regreso"] == "no":
                partes.append("Registré que el servicio es solo ida.")
            elif self.state["regreso"] == "sí":
                partes.append("Anoté que necesitan ida y regreso.")
            else:
                partes.append(f"Tomé nota respecto al regreso: {self.state['regreso']}.")
        if "cantidad" in actualizados:
            partes.append(f"Viajaremos con {self.state['cantidad']} personas.")
        if "especial" in actualizados and self.state["especial"].lower() not in [
            "no",
            "ninguno",
            "ninguna",
            "no tengo",
            "no tenemos",
        ]:
            partes.append(f"Comentario especial: {self.state['especial']}.")
        return " ".join(partes) if partes else "Gracias, actualicé la información."

    def update_state(self, user_input: str):
        new_fields = self.validate_input_llm(user_input)
        # Si la fecha no se extrajo y el usuario solo indicó hora
        if "hora" in new_fields and "fecha" not in new_fields and not self.state["fecha"]:
            # no agregamos fecha vacía; la pregunta siguiente la pedirá
            pass
        # Intenta capturar origen y destino juntos si aún faltan
        if (not self.state["origen"] or not self.state["destino"]) and (
            "origen" not in new_fields or "destino" not in new_fields
        ):
            origen_dest = extract_origin_dest_pair(user_input)
            if origen_dest != (None, None):
                origen_manual, destino_manual = origen_dest
                if origen_manual and "origen" not in new_fields and not self.state["origen"]:
                    new_fields["origen"] = origen_manual
                if destino_manual and "destino" not in new_fields and not self.state["destino"]:
                    new_fields["destino"] = destino_manual
        if "cantidad" not in new_fields and self.state["cantidad"] == 0:
            cantidad_manual = extract_quantity(user_input)
            if cantidad_manual:
                new_fields["cantidad"] = cantidad_manual

        # Si el LLM no extrajo origen/destino, intenta extraerlos manualmente
        if "origen" not in new_fields and not self.state["origen"]:
            origen_manual = extract_location(user_input, "origen")
            if origen_manual:
                new_fields["origen"] = origen_manual
        if "destino" not in new_fields and not self.state["destino"]:
            destino_manual = extract_location(user_input, "destino")
            if destino_manual:
                new_fields["destino"] = destino_manual
        actualizados = {}
        for key, value in new_fields.items():
            if key == "cantidad":
                value_int = extract_quantity(user_input)
                if value_int is None or value_int <= 0:
                    continue
                if self.state[key] != value_int:
                    self.state[key] = value_int
                    actualizados[key] = value_int
            else:
                value_str = str(value).strip()
                if key in {"origen", "destino"}:
                    value_str = normalize_place(value_str)
                    if not value_str or value_str.lower() in LOCATION_EXCLUDE:
                        continue
                if key == "fecha":
                    if not user_mentions_date(user_input):
                        continue
                    value_str = normalize_fecha(value_str, user_input)
                if key == "regreso":
                    if not user_mentions_regreso(user_input):
                        continue
                    normalized_regreso = normalize_regreso(value_str)
                    if not normalized_regreso:
                        continue
                    value_str = normalized_regreso
                if key == "hora":
                    if not user_mentions_time(user_input):
                        continue
                if key == "especial" and value_str.lower() in ["no", "ninguno", "ninguna", "no tengo", "no tenemos"]:
                    self.state[key] = value_str
                    actualizados[key] = value_str
                elif self.state[key] != value_str and value_str:
                    self.state[key] = value_str
                    actualizados[key] = value_str
        if actualizados:
            llm_conf = self._generate_confirmation_llm(actualizados)
            confirm_msg = llm_conf or self._build_confirmation_fallback(actualizados)
            return confirm_msg, True
        return ("", False)

    def run_step(self, user_input: str):
        user_input = user_input.strip()
        if not user_input:
            return "Por favor, escribe tu mensaje."
        mensaje, hubo_cambios = self.update_state(user_input)
        siguiente = self.next_missing()
        final_msg = "¡Gracias! Desde Transportes Miramar revisaremos tu solicitud a la brevedad."
        if hubo_cambios:
            if siguiente:
                pregunta = self.get_next_question()
                return f"{mensaje}\n{pregunta}"
            if not self.saved:
                guardar_cotizacion(self.state)
                self.saved = True
            return f"{mensaje}\n{final_msg}"
        if siguiente:
            if mensaje:
                pregunta = self.get_next_question()
                return f"{mensaje}\n{pregunta}"
            return self.get_next_question()
        if not self.saved:
            guardar_cotizacion(self.state)
            self.saved = True
            if mensaje:
                return f"{mensaje}\n{final_msg}"
            return final_msg
        if mensaje:
            return mensaje
        return "La solicitud ya está registrada con Transportes Miramar. Si necesitas otro traslado, cuéntame."

def main():
    # Carga el modelo y tokenizer una sola vez
    model, tokenizer, device = load_model_and_tokenizer()
    bot = MiramarSellerBot(model, tokenizer, device)

    label_bot = "Transportes Miramar"
    label_user = "Tu"

    saludo_inicial = bot.initial_prompt()
    print(f"{label_bot}: {_sanitize_print(saludo_inicial)}", flush=True)
    cotizando = False

    while True:
        user_input = input(f"\n{label_user}: ").strip()

        # Salida amable
        if re.search(r"\b(chao|adiós|adios|hasta luego|nos vemos|bye)\b", user_input, re.IGNORECASE):
            print(f"{label_bot}: ¡Hasta luego! Si necesitas otra cotización, aquí estaré.")
            break

        # Evita procesar líneas vacías
        if not user_input:
            if cotizando:
                print(f"{label_bot}: Por favor, cuéntame cómo seguimos.")
            else:
                print(f"{label_bot}: ¿Te ayudo con una cotización de viaje?")
            continue

        if cotizando:
            respuesta = bot.run_step(user_input)
            print(f"{label_bot}: {_sanitize_print(respuesta)}", flush=True)
            continue

        # Si aún no se detecta intención de cotizar, el LLM debe decidir
        lower_input = user_input.lower()
        intent = None
        if any(token in lower_input for token in ["cotiz", "presupuesto"]):
            intent = "cotizar"
        if not intent:
            prompt_intent = (BASE_STYLE + f"""
Tarea: Analiza el siguiente mensaje y responde SOLO 'cotizar' si el usuario quiere cotizar un viaje, o 'otro' si no es una solicitud de cotización.
Mensaje del usuario: {user_input}
Respuesta:
""")
            inputs = tokenizer(prompt_intent, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            intent = MiramarSellerBot.decode_new_only(tokenizer, outputs, inputs).strip().lower()

        if "cotizar" in intent:
            cotizando = True
            bot.reset()
            confirm_msg, hubo_cambios = bot.update_state(user_input)
            fragments = [bot.initial_prompt()]
            if hubo_cambios and confirm_msg:
                fragments.append(confirm_msg)
            siguiente = bot.next_missing()
            final_msg = "¡Gracias! Hemos recibido tu solicitud. Un administrador la revisará pronto."
            if siguiente:
                question = bot.get_next_question()
                if question and question not in fragments:
                    fragments.append(question)
            else:
                if not bot.saved:
                    guardar_cotizacion(bot.state)
                    bot.saved = True
                    fragments.append(final_msg)
            if len(fragments) == 1:
                question = bot.get_next_question()
                if question:
                    fragments.append(question)
            response_text = "\n".join(_sanitize_print(part) for part in fragments if part)
            print(f"{label_bot}: {response_text}", flush=True)
            continue

        # Respuesta breve si no es cotización
        prompt_no_cotiza = (BASE_STYLE + f"""
Tarea: Responde de forma cordial y profesional cuando el usuario no está solicitando cotización de viaje.
Guía de estilo: WhatsApp, breve, cálido, sin tecnicismos.
Recuerda: SOLO puedes hablar de cotizaciones de Transportes Miramar. Si el usuario pide otra cosa, explica amablemente que solo gestionas reservas y cotizaciones de Transportes Miramar y ofrece continuar con ese proceso.
Mensaje del usuario: {user_input}
Entrega: <msg>…</msg>
<msg>
""")
        inputs = tokenizer(prompt_no_cotiza, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = MiramarSellerBot.decode_new_only(tokenizer, outputs, inputs)
        m = re.search(r"<msg>(.*?)</msg>", text, flags=re.IGNORECASE | re.DOTALL)
        respuesta_general = m.group(1).strip() if m else text.strip()
        respuesta_general = re.sub(r"</?msg>", "", respuesta_general, flags=re.IGNORECASE).strip()
        print(f"{label_bot}: {_sanitize_print(respuesta_general)}", flush=True)
if __name__ == "__main__":
    main()
