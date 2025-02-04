import sounddevice as sd
from openai import OpenAI
import numpy as np
import soundfile as sf
import os
import subprocess
from dotenv import load_dotenv
import time
from queue import Queue
import asyncio
import concurrent.futures
from google.cloud import speech, texttospeech
import logging
from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
from datetime import datetime
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from asgiref.wsgi import WsgiToAsgi
import requests
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
import threading
from utils import create_interview, get_interview_by_code, update_interview_status
import random
import string
import hashlib
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Flask ve async ayarları
app = Flask(__name__)
CORS(app)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Logger ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# OpenAI istemcisini başlat
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Doğru yol
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(__file__), 'google_credentials.json')

# Gerekli dizinleri oluştur
required_dirs = ['reports', 'temp', 'interview_questions', 'interviews']
for dir_name in required_dirs:
    os.makedirs(dir_name, exist_ok=True)

# E-posta ayarları
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
REPORT_SENDER = os.getenv('REPORT_SENDER')
REPORT_RECIPIENT = os.getenv('REPORT_RECIPIENT')

# Webhook URL'sini güncelle
WEBHOOK_URL = "https://otomasyon.aivatech.io/api/v1/webhooks/B7iYtwVltWEzX2nvAaWCX"

class VoiceAssistant:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.silence_threshold = 0.05
        self.silence_duration = 0.5
        self.audio_queue = Queue()
        self.is_recording = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # API ayarları
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=openai_api_key)

        try:
            self.speech_client = speech.SpeechClient()
            logger.info("Google Cloud Speech client başarıyla başlatıldı")
        except Exception as e:
            logger.error(f"Google Cloud Speech client başlatma hatası: {e}")
            raise

        try:
            self.tts_client = texttospeech.TextToSpeechClient()
            logger.info("Google Cloud Text-to-Speech client başarıyla başlatıldı")
        except Exception as e:
            logger.error(f"Google Cloud Text-to-Speech client başlatma hatası: {e}")
            raise

    def record_audio(self):
        """Ses kaydı yap"""
        logger.debug("Ses kaydı başlıyor...")
        audio_chunks = []
        silence_start = None
        self.is_recording = True

        with sd.InputStream(callback=self.audio_callback,
                          channels=self.channels,
                          samplerate=self.sample_rate):
            while self.is_recording:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.3)
                    audio_chunks.append(audio_chunk)

                    if np.max(np.abs(audio_chunk)) < self.silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.silence_duration:
                            break
                    else:
                        silence_start = None
                except:
                    continue

        logger.debug("Ses kaydı tamamlandı")
        return np.concatenate(audio_chunks) if audio_chunks else None

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Ses kaydı durum: {status}")
        self.audio_queue.put(indata.copy())

    def save_audio(self, recording, filename='temp_recording.wav'):
        logger.debug(f"Ses kaydı kaydediliyor: {filename}")
        sf.write(filename, recording, self.sample_rate)
        return filename

    async def transcribe_audio(self, audio_file):
        try:
            logger.debug("Ses tanıma başlıyor...")
            
            # Ses dosyasını dönüştür
            data, _ = sf.read(audio_file)
            converted_file = 'temp_converted.wav'
            sf.write(converted_file, data, 48000)  # Sample rate'i 48000 olarak ayarla
            
            with open(converted_file, 'rb') as f:
                content = f.read()
                
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=48000,  # Sample rate'i 48000 olarak ayarla
                language_code="tr-TR",
                enable_automatic_punctuation=True,
                use_enhanced=True,
                audio_channel_count=1
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.speech_client.recognize(config=config, audio=audio)
            )
            
            if not response.results:
                logger.warning("Ses tanıma sonuç vermedi")
                return None
                
            transcript = response.results[0].alternatives[0].transcript
            
            # Geçici dosyayı temizle
            if os.path.exists(converted_file):
                os.remove(converted_file)
                
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"Ses tanıma hatası: {str(e)}")
            return None

    async def generate_and_play_speech(self, text):
        """Google TTS ile yanıtı seslendir"""
        try:
            logger.debug("Google TTS ile ses üretimi başlıyor...")

            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="tr-TR",
                name="tr-TR-Standard-A",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=1.0,
                pitch=0,
                volume_gain_db=0.0
            )

            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
            )

            with open("temp_response.wav", "wb") as out:
                out.write(response.audio_content)

            data, fs = sf.read("temp_response.wav")
            sd.play(data, fs)
            sd.wait()

            os.remove("temp_response.wav")

            logger.debug("Google TTS ses üretimi ve oynatma başarılı")
        except Exception as e:
            logger.error(f"Google TTS ses üretme hatası: {str(e)}")

class InterviewAssistant(VoiceAssistant):
    def __init__(self):
        super().__init__()
        self.candidate_name = ""
        self.position = ""
        self.conversation_history = []
        self.sentiment_scores = []
        self.metrics = {
            "iletisim_puani": 0,
            "ozguven_puani": 0,
            "teknik_bilgi": 0,
            "genel_puan": 0
        }
        self.start_time = datetime.now()
        self.interview_questions = []
        self.current_question_index = 0
        
        self.reports_dir = "reports"
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)

    def set_interview_details(self, interview_data):
        """Mülakat detaylarını ayarla"""
        try:
            self.candidate_name = interview_data.get('candidate_name') or interview_data.get('adSoyad')
            self.position = interview_data.get('position') or interview_data.get('isIlaniPozisyonu')
            self.interview_questions = interview_data.get('questions') or interview_data.get('mulakatSorulari')
            
            if not all([self.candidate_name, self.position, self.interview_questions]):
                raise ValueError("Gerekli mülakat bilgileri eksik")
            
            # Hoşgeldin mesajı ve ilk soruyu ekle
            welcome_message = f"Merhaba {self.candidate_name}, {self.position} pozisyonu için mülakatımıza hoş geldiniz. " + self.interview_questions[0]
            self.conversation_history.append({
                "role": "assistant",
                "content": welcome_message
            })
            self.current_question_index = 1
            
        except Exception as e:
            logger.error(f"Mülakat detayları ayarlama hatası: {str(e)}")
            raise

    async def get_gpt_response(self, text):
        """Mülakat bağlamında GPT yanıtını al ve bir sonraki soruyu hazırla"""
        try:
            if not text:
                logger.warning("Boş metin için GPT yanıtı istenemez")
                return None
                
            # İlk yanıt için özel kontrol
            if len(self.conversation_history) == 1:  # Sadece hoşgeldin mesajı varsa
                evaluation = "Hoş geldiniz! Şimdi size ilk sorumu sormak istiyorum."
                next_question = self.interview_questions[0]
                self.current_question_index = 1
            else:
                # Cevabı değerlendir
                evaluation_prompt = f"""
                Aday Yanıtı: {text}
                
                Lütfen bu yanıtı değerlendir ve yapıcı bir geri bildirim ver.
                Yanıt kısa (2-3 cümle) ve motive edici olmalı.
                """
                
                evaluation_response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Sen bir mülakat uzmanısın. Sorularını sırayla sor ve her yanıtı değerlendir."},
                            {"role": "user", "content": evaluation_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=150
                    )
                )
                
                evaluation = evaluation_response.choices[0].message.content
                
                # Bir sonraki soruyu hazırla
                if self.current_question_index < len(self.interview_questions):
                    next_question = self.interview_questions[self.current_question_index]
                    self.current_question_index += 1
                else:
                    next_question = "Mülakat sona erdi. Katılımınız için teşekkür ederiz."
            
            # Konuşma geçmişini güncelle
            self.conversation_history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": evaluation + "\n\n" + next_question}
            ])
            
            return evaluation + "\n\n" + next_question
            
        except Exception as e:
            logger.error(f"GPT yanıt hatası: {str(e)}")
            return None

    async def _analyze_sentiment(self, text):
        """Metni analiz et ve metrikleri güncelle"""
        try:
            sentiment_prompt = f"""
            Lütfen aşağıdaki metni analiz et ve şu metrikleri 0-100 arası puanla:
            Metin: "{text}"
            
            Şu formatta JSON yanıt ver:
            {{
                "iletisim_becerisi": [puan],
                "ozguven": [puan],
                "teknik_bilgi": [puan],
                "aciklama": "[kısa değerlendirme]"
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Sen bir mülakat değerlendirme uzmanısın."},
                        {"role": "user", "content": sentiment_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
            )
            
            # JSON yanıtı parse et
            analysis = json.loads(response.choices[0].message.content)
            
            # Metrikleri güncelle
            self.sentiment_scores.append(analysis)
            
            # Ortalama puanları hesapla
            self.metrics = {
                "iletisim_puani": sum(s["iletisim_becerisi"] for s in self.sentiment_scores) / len(self.sentiment_scores),
                "ozguven_puani": sum(s["ozguven"] for s in self.sentiment_scores) / len(self.sentiment_scores),
                "teknik_bilgi": sum(s["teknik_bilgi"] for s in self.sentiment_scores) / len(self.sentiment_scores)
            }
            self.metrics["genel_puan"] = sum(self.metrics.values()) / 3
            
            logger.debug(f"Metrikler güncellendi: {self.metrics}")
            return analysis["aciklama"]
            
        except Exception as e:
            logger.error(f"Duygu analizi hatası: {str(e)}")
            return None

    async def process_interview_response(self, text):
        """Mülakat yanıtını işle ve değerlendir"""
        try:
            if not text:
                return None
                
            # Bir sonraki soruyu hazırla
            if self.current_question_index < len(self.interview_questions):
                next_question = self.interview_questions[self.current_question_index]
                self.current_question_index += 1
            else:
                next_question = "Mülakat sona erdi."
            
            # Konuşma geçmişini güncelle
            self.conversation_history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": next_question}
            ])
            
            return next_question
            
        except Exception as e:
            logger.error(f"Yanıt işleme hatası: {str(e)}")
            return None

    def generate_pdf_report(self):
        try:
            # PDF dosya yolu
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(self.reports_dir, f"mulakat_raporu_{timestamp}.pdf")
            
            # PDF oluştur
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Başlık
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            story.append(Paragraph("Mülakat Raporu", title_style))
            story.append(Spacer(1, 12))
            
            # Mülakat bilgileri
            info_style = ParagraphStyle(
                'Info',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=6
            )
            story.append(Paragraph(f"Aday: {self.candidate_name}", info_style))
            story.append(Paragraph(f"Pozisyon: {self.position}", info_style))
            story.append(Paragraph(f"Tarih: {self.start_time.strftime('%d.%m.%Y %H:%M')}", info_style))
            story.append(Spacer(1, 20))
            
            # Performans metrikleri
            story.append(Paragraph("Performans Değerlendirmesi", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            metrics_data = [
                ["Metrik", "Puan"],
                ["İletişim Becerisi", f"{self.metrics['iletisim_puani']:.1f}"],
                ["Özgüven", f"{self.metrics['ozguven_puani']:.1f}"],
                ["Teknik Bilgi", f"{self.metrics['teknik_bilgi']:.1f}"],
                ["Genel Puan", f"{self.metrics['genel_puan']:.1f}"]
            ]
            
            t = Table(metrics_data, colWidths=[300, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
            story.append(Spacer(1, 20))
            
            # Konuşma geçmişi
            story.append(Paragraph("Mülakat Detayları", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for entry in self.conversation_history:
                if entry["role"] == "user":
                    story.append(Paragraph(f"<b>Aday:</b> {entry['content']}", styles['Normal']))
                else:
                    story.append(Paragraph(f"<b>Mülakat Uzmanı:</b> {entry['content']}", styles['Normal']))
                story.append(Spacer(1, 12))
            
            # PDF oluştur
            doc.build(story)
            logger.info(f"PDF raporu başarıyla oluşturuldu: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"PDF rapor oluşturma hatası: {str(e)}")
            return None

    def send_report_email(self, pdf_path):
        try:
            # E-posta oluştur
            msg = MIMEMultipart()
            msg['From'] = REPORT_SENDER
            msg['To'] = REPORT_RECIPIENT
            msg['Subject'] = f"Mülakat Raporu - {self.candidate_name} - {self.position}"
            
            # E-posta metni
            body = f"""
            Merhaba,
            
            {self.candidate_name} adayı ile {self.position} pozisyonu için yapılan mülakat raporu ekte yer almaktadır.
            
            Mülakat Bilgileri:
            - Aday: {self.candidate_name}
            - Pozisyon: {self.position}
            - Tarih: {self.start_time.strftime('%d.%m.%Y %H:%M')}
            - Genel Puan: {self.metrics['genel_puan']:.1f}/100
            
            Detaylı değerlendirme için ekteki PDF dosyasını inceleyebilirsiniz.
            
            İyi çalışmalar,
            AIVA Mülakat Asistanı
            """
            msg.attach(MIMEText(body, 'plain'))
            
            # PDF ekle
            with open(pdf_path, "rb") as f:
                pdf = MIMEApplication(f.read(), _subtype="pdf")
                pdf.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
                msg.attach(pdf)
            
            # E-postayı gönder
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)

            # Webhook'a rapor gönder
            self.send_report_webhook(pdf_path)
            
            return True
        except Exception as e:
            print(f"E-posta gönderme hatası: {str(e)}")
            return False

    def send_report_webhook(self, pdf_path):
        try:
            # Konuşma geçmişini düzgün şekilde hazırla
            conversation_flow = []
            for entry in self.conversation_history:
                conversation_flow.append({
                    "soru_cevap": entry["content"] if entry["role"] == "user" else "",
                    "degerlendirme": entry["content"] if entry["role"] == "assistant" else ""
                })
                
                # Webhook verisi
                webhook_data = {
                    "aday_bilgileri": {
                        "isim": self.candidate_name,
                        "pozisyon": self.position,
                        "tarih": self.start_time.strftime('%d.%m.%Y %H:%M')
                    },
                    "metrikler": self.metrics,
                    "konusma_akisi": conversation_flow
                }
                
            # Webhook isteği gönder
            response = requests.post(
                WEBHOOK_URL,
                json=webhook_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                logger.error(f"Webhook hatası: {response.status_code} - {response.text}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Webhook gönderme hatası: {str(e)}")
            return False

    def _prepare_interview_questions(self):
        """Pozisyona göre mülakat sorularını hazırla"""
        try:
            prompt = f"""
            {self.position} pozisyonu için 5 adet mülakat sorusu hazırla. Her soru teknik bilgi ve deneyimi ölçmeye yönelik olmalı.
            
            Soruları JSON formatında döndür:
            {{"sorular": ["soru1", "soru2", ...]}}
            
            Her soru profesyonel ve nazik bir dille sorulmalı.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen deneyimli bir İK uzmanısın. Mülakatları profesyonel ve yapıcı bir şekilde yönetirsin."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            questions = json.loads(response.choices[0].message.content)
            self.interview_questions = questions["sorular"]
            
            # İlk mesajı ekle
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Merhaba {self.candidate_name}, {self.position} pozisyonu için mülakatımıza hoş geldiniz. Nasılsınız?"
            })
            
            logger.info(f"Mülakat soruları hazırlandı: {len(self.interview_questions)} soru")
            
        except Exception as e:
            logger.error(f"Soru hazırlama hatası: {str(e)}")
            # Varsayılan sorular
            self.interview_questions = [
                "Kendinizden ve kariyerinizden bahseder misiniz?",
                "Bu pozisyona neden başvurdunuz?",
                "Önceki iş deneyimlerinizden bahseder misiniz?",
                "Teknik becerileriniz nelerdir?",
                "Gelecekteki kariyer hedefleriniz nelerdir?"
            ]
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Merhaba {self.candidate_name}, {self.position} pozisyonu için mülakatımıza hoş geldiniz. Nasılsınız?"
            })

# Global değişken
current_interview = None

@app.route('/')
def home():
    return render_template('create_interview.html')

@app.route('/join')
def join():
    return render_template('interview_entry.html')

@app.route('/interview')
def interview():
    code = request.args.get('code')
    if not code:
        return render_template('interview_entry.html', error='Mülakat kodu gereklidir')

    interview_data = get_interview_by_code(code)
    if not interview_data:
        return render_template('interview_entry.html', error='Geçersiz mülakat kodu')

    if interview_data.get('status') == 'completed':
        return render_template('interview_entry.html', error='Bu mülakat tamamlanmış')

    # Mülakat asistanını başlat
    global current_interview
    current_interview = InterviewAssistant()
    current_interview.set_interview_details(interview_data)

    update_interview_status(code, 'in_progress')
    return render_template('interview.html', interview=interview_data)

@app.route('/create_interview', methods=['POST'])
def handle_create_interview():
    try:
        data = request.get_json()
        candidate_name = data.get('candidate_name')
        position = data.get('position')

        if not candidate_name or not position:
            return jsonify({
                'success': False,
                'error': 'Aday adı ve pozisyon gereklidir'
            })

        interview_code = create_interview(candidate_name, position)
        return jsonify({
            'success': True,
            'code': interview_code
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/verify_code', methods=['POST'])
def verify_code():
    try:
        data = request.get_json()
        code = data.get('code')

        if not code:
            return jsonify({
                'success': False,
                'error': 'Mülakat kodu gereklidir'
            })

        interview_data = get_interview_by_code(code)
        if interview_data:
            return jsonify({'success': True})
        else:
            return jsonify({
                'success': False,
                'error': 'Geçersiz mülakat kodu'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/start_interview', methods=['POST'])
async def start_interview():
    try:
        data = request.json
        if not data or 'candidate_name' not in data or 'position' not in data:
            return jsonify({
                "success": False,
                "error": "Aday adı ve pozisyon bilgisi gerekli"
            }), 400

        global current_interview
        current_interview = InterviewAssistant()
        current_interview.set_interview_details(
            data['candidate_name'],
            data['position']
        )
        
        # Soruları hazırla ve kaydet
        interview_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        questions_file = os.path.join('interviews', f'{interview_code}.json')
        
        if not os.path.exists('interviews'):
            os.makedirs('interviews')
            
        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump({
                'candidate_name': data['candidate_name'],
                'position': data['position'],
                'questions': current_interview.interview_questions
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Mülakat başlatıldı: {data['candidate_name']} - {data['position']}")
        
        return jsonify({
            "success": True,
            "message": "Mülakat başarıyla başlatıldı",
            "code": interview_code
        })
        
    except Exception as e:
        logger.error(f"Mülakat başlatma hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Mülakat başlatılamadı: {str(e)}"
        }), 500

@app.route('/start_recording', methods=['POST'])
async def start_recording():
    try:
        if not current_interview:
            return jsonify({
                "success": False,
                "error": "Lütfen önce mülakatı başlatın"
            }), 400
            
        # Ses kaydını başlat
        recording = current_interview.record_audio()
        if recording:
            filename = current_interview.save_audio(recording)
            return jsonify({
                "success": True,
                "message": "Ses kaydı başarıyla tamamlandı",
                "filename": filename
            })
        else:
            return jsonify({
                "success": False,
                "error": "Ses kaydı alınamadı"
            }), 400
        
    except Exception as e:
        logger.error(f"Ses kaydı başlatma hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/process_audio', methods=['POST'])
async def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({
                "success": False,
                "error": "Ses dosyası bulunamadı",
                "continue_listening": True
            }), 400
            
        audio_file = request.files['audio']
        
        # Geçici dosya yolu oluştur
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        temp_path = os.path.join(temp_dir, f'temp_audio_{time.time()}.wav')
        webm_path = temp_path + '.webm'
        
        try:
            # WebM dosyasını kaydet
            audio_file.save(webm_path)
            
            # FFmpeg komutu
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-i', webm_path,
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                temp_path
            ]
            
            subprocess.run(ffmpeg_command, check=True, capture_output=True)
            
            # Ses seviyesi analizi
            data, _ = sf.read(temp_path)
            volume_level = float(np.max(np.abs(data)) * 100)
            
            # Sessizlik kontrolü
            is_silence = bool(volume_level < 5)
            
            # Google Speech client
            client = speech.SpeechClient()
            
            with open(temp_path, 'rb') as audio_file:
                content = audio_file.read()
            
            # Ses tanıma yapılandırması
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="tr-TR",
                enable_automatic_punctuation=True,
                use_enhanced=True,
                audio_channel_count=1
            )
            
            # Ses tanıma işlemi
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.recognize(config=config, audio=audio)
            )
            
            if not response.results or is_silence:
                return jsonify({
                    "success": False,
                    "error": "Sizi duyamadım, lütfen tekrar konuşun",
                    "continue_listening": True,
                    "volume_level": float(volume_level),
                    "should_restart": True  # Yeni eklenen alan
                }), 400
                
            transcript = response.results[0].alternatives[0].transcript
            confidence = float(response.results[0].alternatives[0].confidence)
            
            if confidence < 0.6:
                return jsonify({
                    "success": False,
                    "error": "Sizi net anlayamadım, lütfen tekrar konuşun",
                    "continue_listening": True,
                    "volume_level": float(volume_level),
                    "confidence": float(confidence),
                    "should_restart": True  # Yeni eklenen alan
                }), 400
            
            # Mevcut mülakat mantığı
            if current_interview:
                response = await current_interview.process_interview_response(transcript)
                if not response:
                    return jsonify({
                        "success": False,
                        "error": "Lütfen tekrar konuşun",
                        "continue_listening": True,
                        "volume_level": float(volume_level),
                        "should_restart": True  # Yeni eklenen alan
                    }), 400
                
                interview_completed = bool(current_interview.current_question_index >= len(current_interview.interview_questions))
                
                return jsonify({
                    "success": True,
                    "transcript": transcript,
                    "response": response,
                    "volume_level": float(volume_level),
                    "confidence": float(confidence),
                    "continue_listening": True,
                    "interview_completed": interview_completed,
                    "should_restart": True  # Yeni eklenen alan
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Mülakat henüz başlatılmadı",
                    "continue_listening": False
                }), 400
                
        finally:
            # Geçici dosyaları temizle
            for file_path in [webm_path, temp_path]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Dosya silme hatası ({file_path}): {str(e)}")
                
    except Exception as e:
        logger.error(f"Ses işleme hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Lütfen tekrar konuşun",
            "continue_listening": True,
            "should_restart": True  # Yeni eklenen alan
        }), 500

@app.route('/generate_report', methods=['POST'])
async def generate_report():
    try:
        if not current_interview:
            return jsonify({
                "success": False,
                "error": "Mülakat oturumu bulunamadı"
            }), 400

        # Konuşma geçmişini al
        conversation_history = request.json.get('conversation_history', [])
        
        # GPT ile değerlendirme yap
        evaluation_prompt = f"""
        Aşağıdaki mülakat konuşmasını değerlendir ve bir rapor oluştur:
        
        Aday: {current_interview.candidate_name}
        Pozisyon: {current_interview.position}
        
        Konuşma Geçmişi:
        {json.dumps(conversation_history, indent=2, ensure_ascii=False)}
        
        Lütfen aşağıdaki başlıklara göre değerlendirme yap:
        1. Teknik Bilgi ve Deneyim
        2. İletişim Becerileri
        3. Problem Çözme Yeteneği
        4. Genel Değerlendirme
        5. Öneriler
        """
        
        evaluation_response = await asyncio.get_event_loop().run_in_executor(
            current_interview.executor,
            lambda: current_interview.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen bir mülakat değerlendirme uzmanısın."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
        )
        
        evaluation = evaluation_response.choices[0].message.content
        
        # PDF raporu oluştur
        pdf_path = current_interview.generate_pdf_report()
        if not pdf_path:
            return jsonify({
                "success": False,
                "error": "PDF raporu oluşturulamadı"
            }), 500
        
        # Raporu e-posta ile gönder
        if current_interview.send_report_email(pdf_path):
            # Webhook'a gönder
            current_interview.send_report_webhook(pdf_path)
            
            return jsonify({
                "success": True,
                "message": "Rapor oluşturuldu ve gönderildi",
                "evaluation": evaluation
            })
        else:
            return jsonify({
                "success": False,
                "error": "Rapor gönderilemedi"
            }), 500
            
    except Exception as e:
        logger.error(f"Rapor oluşturma hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/check_audio_support', methods=['GET'])
def check_audio_support():
    """Desteklenen ses formatlarını kontrol et"""
    try:
        supported_formats = {
            'webm': True,
            'wav': True,
            'mp3': True,
            'ogg': True
        }
        return jsonify({
            'success': True,
            'supported_formats': supported_formats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Veritabanı bağlantısı için yardımcı fonksiyon
def get_db_connection():
    try:
        conn = sqlite3.connect('data/interview.db')
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Veritabanı bağlantı hatası: {str(e)}")
        raise

def create_or_get_interview_code(email):
    """Email adresine göre mülakat kodu oluştur veya var olanı getir"""
    conn = get_db_connection()
    try:
        # Önce mevcut kodu kontrol et
        cursor = conn.execute('SELECT code FROM interview_codes WHERE email = ?', (email,))
        result = cursor.fetchone()
        
        if result:
            return result['code']
            
        # Yeni kod oluştur
        while True:
            # Email'den benzersiz bir kod oluştur
            hash_object = hashlib.md5(email.encode())
            code = hash_object.hexdigest()[:6].upper()
            
            # Kodun benzersiz olduğunu kontrol et
            cursor = conn.execute('SELECT code FROM interview_codes WHERE code = ?', (code,))
            if not cursor.fetchone():
                break
        
        # Yeni kodu kaydet
        conn.execute('INSERT INTO interview_codes (email, code, created_at) VALUES (?, ?, ?)',
                    (email, code, datetime.now().isoformat()))
        conn.commit()
        return code
    finally:
        conn.close()

def save_interview_data(data, code):
    """Mülakat verilerini JSON dosyası olarak kaydet"""
    try:
        # Veriyi yeni formata dönüştür
        formatted_data = {
            "code": code,
            "candidate_name": data.get('adSoyad') or data.get('candidate_name'),
            "position": data.get('isIlaniPozisyonu') or data.get('position'),
            "questions": data.get('mulakatSorulari') or data.get('questions', [
                "1. Yapay zekanın temel bileşenleri hakkında bilgi verebilir misiniz?",
                "2. Belirli bir veri setinde overfitting problemini nasıl tanımlar ve çözersiniz?",
                "3. Günlük çalışmalarınızda hangi yapay zeka frameworklerini kullandınız?",
                "4. Çeşitli regresyon teknikleri hakkında bilgi verebilir misiniz?",
                "5. NLP konusunda ne gibi deneyimleriniz var?"
            ]),
            "status": data.get('status', 'pending'),
            "created_at": data.get('created_at', datetime.now().isoformat()),
            "updated_at": data.get('updated_at', datetime.now().isoformat())
        }
        
        file_path = os.path.join('interviews', f'{code}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Mülakat verisi kaydedildi: {code}")
        return file_path
    except Exception as e:
        logger.error(f"JSON kaydetme hatası: {str(e)}")
        return None

def send_webhook_notification(webhook_url, data):
    """Webhook'a bildirim gönder"""
    try:
        response = requests.post(
            webhook_url,
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Webhook gönderme hatası: {str(e)}")
        return False

@app.route('/webhook/interview', methods=['POST'])
def webhook_interview_handler():
    try:
        data = request.get_json()
        logger.info(f"Gelen webhook verisi: {data}")
        
        # Webhook verilerini kontrol et
        if not data or 'adSoyad' not in data or 'mail' not in data:
            logger.error("Geçersiz webhook verisi")
            return jsonify({
                'success': False,
                'error': 'Geçersiz veri formatı'
            }), 400
        
        # Mülakat kodu oluştur
        code = create_or_get_interview_code(data['mail'])
        
        # Mülakat verilerini hazırla
        interview_data = {
            "code": code,
            "candidate_name": data['adSoyad'],
            "position": data.get('isIlaniPozisyonu', 'Genel Pozisyon'),
            "questions": data.get('questions', [
                "1. Yapay zekanın temel bileşenleri hakkında bilgi verebilir misiniz?",
                "2. Belirli bir veri setinde overfitting problemini nasıl tanımlar ve çözersiniz?",
                "3. Günlük çalışmalarınızda hangi yapay zeka frameworklerini kullandınız?",
                "4. Çeşitli regresyon teknikleri hakkında bilgi verebilir misiniz?",
                "5. NLP konusunda ne gibi deneyimleriniz var?"
            ]),
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # JSON dosyasını kaydet
        save_interview_data(interview_data, code)
        logger.info(f"Mülakat verisi kaydedildi: {code}")
        
        # Mail gönder
        try:
            msg = MIMEMultipart()
            msg['From'] = SMTP_USERNAME
            msg['To'] = data['mail']
            msg['Subject'] = "Mülakat Kodunuz"
            
            body = f"""
            Merhaba {data['adSoyad']},
            
            Mülakat kodunuz: {code}
            
            Bu kod ile mülakata katılabilirsiniz.
            Mülakat sistemine giriş yapmak için bu kodu kullanın.
            
            İyi çalışmalar,
            AIVA Mülakat Sistemi
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
                
            logger.info(f"Mülakat kodu mail ile gönderildi: {data['mail']}")
            
        except Exception as e:
            logger.error(f"Mail gönderme hatası: {str(e)}")
        
        return jsonify({
            'success': True,
            'code': code,
            'message': 'Mülakat oluşturuldu ve mail gönderildi'
        })
        
    except Exception as e:
        logger.error(f"Webhook işleme hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    try:
        # Veritabanını hazırla
        init_db()
        
        # Dosya izleme sistemini başlat
        observer = start_file_watcher()
        
        # Flask uygulamasını başlat
        app.run(host='0.0.0.0', port=5004)
        
    except Exception as e:
        logger.critical(f"Program başlatılamadı: {str(e)}")
    finally:
        if 'observer' in locals() and observer:
            observer.stop()
            observer.join()
            
            
            
            
        
    
    
        
        
            



    