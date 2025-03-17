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
import logging
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, Response, session, url_for, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from asgiref.wsgi import WsgiToAsgi
import requests
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import string
import hashlib
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import secrets
import aiohttp
import queue
import playsound
import wave
import tempfile
from functools import wraps
import re
import base64
from flask_session import Session

# .env dosyasını yükle
load_dotenv()

# Logging seviyesini ayarla - DEBUG yerine INFO kullan
logging.basicConfig(level=logging.INFO)

# Flask ve async ayarları
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = False  # Development için False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.getenv('SECRET_KEY', 'aiva_secret_key_2024')
Session(app)

# CORS ayarları
CORS(app, supports_credentials=True)

# Logging ayarları
logger = logging.getLogger(__name__)

# OpenAI istemcisini başlat
openai_client = OpenAI()

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

# Webhook URL'lerini güncelle
WEBHOOK_ADAY_URL = os.getenv('WEBHOOK_ADAY_URL')
WEBHOOK_RAPOR_URL = os.getenv('WEBHOOK_RAPOR_URL')

# Global değişkenler
interview_data = {}  # Mülakat verilerini saklamak için
SILENCE_THRESHOLD = 0.05  # Sessizlik eşik değeri
VOICE_THRESHOLD = 0.15   # Ses algılama eşik değeri
SILENCE_DURATION = 1500  # Sessizlik süresi (ms)
MIN_CONFIDENCE = 0.6     # Minimum güven skoru

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
        logger.info("OpenAI client başarıyla başlatıldı")

    def record_audio(self):
        """Ses kaydı yap"""
        logger.debug("Ses kaydı başlıyor...")
        audio_chunks = []
        silence_start = None
        self.is_recording = True
        
        # Ses algılama parametreleri
        SILENCE_THRESHOLD = 0.02  # Sessizlik eşiği
        MIN_AUDIO_LENGTH = 0.3    # Minimum ses süresi (saniye)
        MAX_SILENCE_LENGTH = 1.5  # Maksimum sessizlik süresi (saniye)
        BUFFER_SIZE = 1024        # Tampon boyutu
        
        try:
            with sd.InputStream(callback=self.audio_callback,
                              channels=self.channels,
                              samplerate=self.sample_rate,
                              blocksize=BUFFER_SIZE,
                              device=None):  # Varsayılan ses cihazı
                
                last_active_time = time.time()
                
                while self.is_recording:
                    try:
                        audio_chunk = self.audio_queue.get(timeout=0.1)
                        current_volume = np.max(np.abs(audio_chunk))
                        
                        # Ses algılandı
                        if current_volume > SILENCE_THRESHOLD:
                            last_active_time = time.time()
                            if silence_start is not None:
                                silence_duration = time.time() - silence_start
                                if silence_duration > MAX_SILENCE_LENGTH:
                                    audio_chunks = []  # Tampon temizlenir
                                silence_start = None
                            audio_chunks.append(audio_chunk)
                        # Sessizlik algılandı
                        else:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > MAX_SILENCE_LENGTH:
                                # Yeterli uzunlukta ses kaydı varsa bitir
                                if len(audio_chunks) * (BUFFER_SIZE / self.sample_rate) >= MIN_AUDIO_LENGTH:
                                    break
                            
                        # Uzun süre ses algılanmazsa kaydı durdur
                        if time.time() - last_active_time > 5:  # 5 saniye sessizlik
                            if len(audio_chunks) > 0:
                                break
                            else:
                                audio_chunks = []  # Tampon temizle
                                last_active_time = time.time()  # Zamanı sıfırla
                                
                    except queue.Empty:
                        continue
                        
        except Exception as e:
            logger.error(f"Ses kaydı hatası: {str(e)}")
            return None
            
        logger.debug("Ses kaydı tamamlandı")
        return np.concatenate(audio_chunks) if audio_chunks else None

    def audio_callback(self, indata, frames, time, status):
        """Ses verisi callback fonksiyonu"""
        if status:
            logger.warning(f"Ses kaydı durum: {status}")
        try:
            self.audio_queue.put(indata.copy())
        except queue.Full:
            self.audio_queue.get_nowait()  # En eski veriyi at
            self.audio_queue.put(indata.copy())

    def save_audio(self, recording, filename='temp_recording.wav'):
        logger.debug(f"Ses kaydı kaydediliyor: {filename}")
        sf.write(filename, recording, self.sample_rate)
        return filename

    async def transcribe_audio(self, audio_file):
        try:
            logger.debug("Ses tanıma başlıyor...")
            
            # WebM'den WAV'a dönüştür
            converted_file = 'temp/temp_converted.wav'
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-i', audio_file,
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '48000',
                converted_file
            ]
            
            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True)
                logger.info("WebM dosyası WAV formatına dönüştürüldü")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg dönüştürme hatası: {e.stderr.decode()}")
                return None

            # OpenAI Whisper API kullanarak ses tanıma
            with open(converted_file, 'rb') as audio:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language="tr"
                )

            # Geçici dosyaları temizle
            for temp_file in [converted_file]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.info(f"Geçici dosya silindi: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Geçici dosya silme hatası: {str(e)}")
            
            return transcript.text.strip()
            
        except Exception as e:
            logger.error(f"Ses tanıma hatası: {str(e)}")
            return None

    async def generate_and_play_speech(self, text):
        """OpenAI TTS ile metni sese çevirir ve oynatır"""
        try:
            # Tuple ise ilk elemanı al, değilse direkt metni kullan
            if isinstance(text, tuple):
                text = text[0]
                
            logger.info(f"OpenAI TTS ile ses oluşturuluyor... Metin: {text[:50]}...")
            
            # Temp klasörünü kontrol et
            if not os.path.exists('temp'):
                os.makedirs('temp')
                
            speech_file_path = "temp/temp_speech.mp3"
            
            # OpenAI TTS API çağrısı
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            
            # Ses dosyasını kaydet
            response.stream_to_file(speech_file_path)
            
            # Sesi oynat
            playsound.playsound(speech_file_path)
            
            # Geçici dosyayı sil
            if os.path.exists(speech_file_path):
                os.remove(speech_file_path)
                logger.info("Geçici ses dosyası silindi")
                
            return True
                
        except Exception as e:
            logger.error(f"OpenAI TTS API hatası: {e}")
            return False

   

class InterviewAssistant(VoiceAssistant):
    def __init__(self):
        super().__init__()
        self.code = None
        self.candidate_name = ""
        self.position = ""
        self.requirements = []
        self.custom_questions = []
        self.cv_summary = ""
        self.pre_info = ""
        self.conversation_history = []
        self.sentiment_scores = []
        self.metrics = {
            "iletisim_puani": 0,
            "ozguven_puani": 0,
            "teknik_bilgi": 0,
            "genel_puan": 0
        }
        self.start_time = datetime.now()
        self.current_question_index = 0
        self.interview_questions = []

    def set_interview_details(self, code):
        """Mülakat detaylarını JSON dosyasından ayarla"""
        try:
            # JSON dosyasından mülakat bilgilerini oku
            json_path = os.path.join('interviews', f'{code}.json')
            if not os.path.exists(json_path):
                raise ValueError(f"Mülakat dosyası bulunamadı: {code}")

            with open(json_path, 'r', encoding='utf-8') as f:
                interview_data = json.load(f)

            # Mülakat bilgilerini ayarla
            self.code = code
            self.candidate_name = interview_data.get("candidate_name", "")
            self.position = interview_data.get("position", "")
            self.requirements = interview_data.get("requirements", [])
            self.custom_questions = interview_data.get("questions", [])
            self.cv_summary = interview_data.get("cv_summary", "")
            self.pre_info = interview_data.get("pre_info", "")

            # Soruları hazırla
            self._prepare_interview_questions()

            logger.info(f"Mülakat detayları ayarlandı: {code}")
            return True

        except Exception as e:
            logger.error(f"Mülakat detayları ayarlama hatası: {str(e)}")
            raise

    def _prepare_interview_questions(self):
        """Özelleştirilmiş mülakat sorularını hazırla"""
        try:
            # Özel soruları kullan
            self.interview_questions = self.custom_questions

            logger.info(f"Mülakat soruları hazırlandı: {len(self.interview_questions)} soru")

        except Exception as e:
            logger.error(f"Soru hazırlama hatası: {str(e)}")
            # Varsayılan sorular
            self.interview_questions = [
                "Kendinizden ve kariyerinizden bahseder misiniz?",
                "Bu pozisyona neden başvurdunuz?",
                "Önceki iş deneyimlerinizden bahseder misiniz?",
                "Teknik becerileriniz nelerdir?",
                "Zorlu bir iş durumunu nasıl çözdüğünüzü anlatır mısınız?",
                "Gelecekteki kariyer hedefleriniz nelerdir?"
            ]

    async def get_gpt_response(self, text):
        try:
            # Başlangıç kontrolü
            if not hasattr(self, 'interview_started'):
                if any(word in text.lower() for word in ["başlayalım", "başla", "hazırım", "başlayabiliriz"]):
                    self.interview_started = True
                    self.current_question_index = 0
                    welcome_message = f"""Merhaba {self.candidate_name}, {self.position} pozisyonu için mülakatımıza hoş geldiniz.

Özgeçmişinizde belirttiğiniz bilgilere dayanarak, pozisyonun gerektirdiği yetkinlikler hakkında konuşacağız.

İlk olarak, {self.interview_questions[self.current_question_index]}"""
                    return welcome_message, False
                else:
                    return "Mülakata başlamak için 'Başlayalım' diyebilirsiniz.", False

            # Tekrar isteği kontrolü
            if any(phrase in text.lower() for phrase in ["tekrar", "anlamadım", "tekrarlar mısın", "bir daha söyler misin"]):
                return f"Tabii ki, soruyu tekrar ediyorum: {self.interview_questions[self.current_question_index]}", False

            # Kısa cevap kontrolü
            if len(text.split()) < 5:
                follow_up_responses = [
                    "Bu konu hakkında biraz daha detay verebilir misiniz?",
                    "İlginç, peki bu konuda başka neler söylemek istersiniz?",
                    "Deneyimlerinizden örnekler verebilir misiniz?",
                    "Bu konuyu biraz daha açar mısınız?"
                ]
                return random.choice(follow_up_responses), False

            # Bir sonraki soruya geçiş
            if self.current_question_index < len(self.interview_questions) - 1:
                self.current_question_index += 1
                transition_phrases = [
                    f"Anlıyorum, teşekkür ederim. Şimdi başka bir konuya geçelim. {self.interview_questions[self.current_question_index]}",
                    f"Bu konudaki görüşleriniz için teşekkürler. Peki, {self.interview_questions[self.current_question_index]}",
                    f"Güzel bir açıklama oldu. İsterseniz şimdi {self.interview_questions[self.current_question_index]}",
                    f"Teşekkür ederim. Bir sonraki konumuz: {self.interview_questions[self.current_question_index]}"
                ]
                return random.choice(transition_phrases), False
            else:
                # Son soruya gelindi, mülakat kapanış mesajı
                closing_message = """Mülakat sorularımız tamamlandı. Paylaştığınız değerli bilgiler ve ayırdığınız zaman için teşekkür ederiz. 
                
Değerlendirme sonucunu en kısa sürede size ileteceğiz. İyi günler dilerim."""
                # İkinci parametre True olarak döndürülüyor, bu mülakat bittiğini gösterir
                return closing_message, True

        except Exception as e:
            logger.error(f"GPT yanıtı alma hatası: {str(e)}")
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar dener misiniz?", False

    def prepare_conversation_context(self, text):
        return [
            {"role": "system", "content": f"Sen bir mülakat uzmanısın. {self.position} pozisyonu için mülakat yapıyorsun."},
            {"role": "user", "content": text}
        ]

    async def _analyze_sentiment(self, text):
        """Adayın cevaplarını analiz et"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Verilen metni analiz et ve duygusal durumu değerlendir."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
            )
            
            sentiment = response.choices[0].message.content
            logger.info(f"Duygu analizi sonucu: {sentiment}")
            return sentiment
            
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
            pdf_path = os.path.join('reports', f"mulakat_raporu_{timestamp}.pdf")
            
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
            story.append(Paragraph(f"Aday: {self.candidate_name if current_interview else 'Bilinmiyor'}", info_style))
            story.append(Paragraph(f"Pozisyon: {self.position if current_interview else 'Bilinmiyor'}", info_style))
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

    def send_report_webhook(self, pdf_path, evaluation_data):
        """Raporu webhook'a gönder"""
        try:
            # PDF dosya adını URL'ye dönüştür
            pdf_filename = os.path.basename(pdf_path)
            report_url = url_for('serve_report', filename=pdf_filename, _external=True)
            
            # Mülakat verilerini hazırla
            interview_summary = {
                "mulakat_kodu": self.code,
                "aday_bilgileri": {
                    "isim": self.candidate_name,
                    "pozisyon": self.position,
                    "cv_ozeti": self.cv_summary,
                    "on_bilgi": self.pre_info,
                    "gereksinimler": self.requirements,
                    "tarih": self.start_time.isoformat()
                },
                "mulakat_metrikleri": {
                    "iletisim_puani": self.metrics["iletisim_puani"],
                    "ozguven_puani": self.metrics["ozguven_puani"],
                    "teknik_bilgi": self.metrics["teknik_bilgi"],
                    "genel_puan": self.metrics["genel_puan"]
                },
                "degerlendirme": evaluation_data,
                "konusma_akisi": self.conversation_history,
                "rapor_url": report_url,  # PDF dosyasının kendisi yerine URL'si gönderiliyor
                "olusturulma_tarihi": datetime.now().isoformat()
            }
            
            # JSON dosyasından webhook URL'sini kontrol et
            json_path = os.path.join('interviews', f'{self.code}.json')
            webhook_url = WEBHOOK_RAPOR_URL  # Varsayılan URL
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    interview_json = json.load(f)
                    if 'webhook_rapor_url' in interview_json:
                        webhook_url = interview_json['webhook_rapor_url']
                        logger.info(f"JSON'dan webhook URL'si kullanılıyor: {webhook_url}")
                    else:
                        logger.info(f"Varsayılan webhook URL'si kullanılıyor: {webhook_url}")
            
            # WebhookRapor'a gönder
            response = requests.post(
                webhook_url,
                json=interview_summary,
                headers={
                    'Content-Type': 'application/json',
                    'X-Interview-Code': self.code
                }
            )
            
            if response.status_code == 200:
                logger.info(f"Rapor webhook'a başarıyla gönderildi: {self.code}")
                return True
            else:
                logger.error(f"Rapor webhook'a gönderilemedi: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Webhook gönderme hatası: {str(e)}")
            return False

@app.route('/reports/<filename>')
def serve_report(filename):
    """Raporları sunmak için route"""
    try:
        # Güvenlik kontrolü - sadece pdf dosyalarına izin ver ve path traversal saldırılarını engelle
        if not filename.endswith('.pdf') or '..' in filename or '/' in filename:
            return 'Geçersiz dosya ismi', 400
            
        # Reports klasörünün varlığını kontrol et
        if not os.path.exists('reports'):
            os.makedirs('reports', exist_ok=True)
            
        # Dosya yolunu oluştur
        file_path = os.path.join('reports', filename)
        
        # Dosyanın varlığını kontrol et
        if not os.path.exists(file_path):
            return 'Rapor bulunamadı', 404
            
        # PDF dosyasını gönder
        return send_file(file_path, 
                         mimetype='application/pdf',
                         as_attachment=True,
                         download_name=filename)
    except Exception as e:
        logger.error(f"Rapor sunma hatası: {str(e)}")
        return 'Rapor sunulurken bir hata oluştu', 500

def start_file_watcher():
    """Dosya izleme sistemini başlat - Sadece terminal log'ları için"""
    try:
        class InterviewFileHandler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    if event.src_path.endswith('.json'):
                        print(f"\n[+] Yeni mülakat dosyası oluşturuldu: {os.path.basename(event.src_path)}")
                    elif event.src_path.endswith('.pdf'):
                        print(f"\n[+] Yeni rapor oluşturuldu: {os.path.basename(event.src_path)}")
                    
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.json'):
                    print(f"\n[*] Mülakat dosyası güncellendi: {os.path.basename(event.src_path)}")
                    
            def on_deleted(self, event):
                if not event.is_directory:
                    if event.src_path.endswith('.json'):
                        print(f"\n[-] Mülakat dosyası silindi: {os.path.basename(event.src_path)}")
                    elif event.src_path.endswith('.pdf'):
                        print(f"\n[-] Rapor silindi: {os.path.basename(event.src_path)}")

        # İzlenecek dizinler
        paths_to_watch = ['interviews', 'reports']
        
        # Dizinleri oluştur
        for path in paths_to_watch:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"\n[+] {path} dizini oluşturuldu")
        
        # Observer oluştur ve başlat
        observer = Observer()
        for path in paths_to_watch:
            observer.schedule(InterviewFileHandler(), path, recursive=False)
        
        observer.start()
        print("\n[+] Dosya izleme sistemi başlatıldı")
        print("[*] interviews/ ve reports/ dizinleri izleniyor...")
        
        return observer
        
    except Exception as e:
        print(f"\n[!] Dosya izleme sistemi başlatılamadı: {str(e)}")
        return None

async def get_realtime_token():
    """OpenAI API anahtarını döndür"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY bulunamadı")
            return None
            
        return api_key
                
    except Exception as e:
        logger.error(f"Token alma hatası: {str(e)}")
        return None

@app.route('/get_interview_token', methods=['POST'])
async def get_interview_token():
    try:
        data = request.get_json()
        interview_code = data.get('code')
        
        if not interview_code:
            return jsonify({
                'success': False,
                'error': 'Mülakat kodu gerekli'
            })
            
        token = await get_realtime_token()
        if not token:
            return jsonify({
                'success': False,
                'error': 'Token alınamadı'
            })
            
        # Benzersiz bir session_id oluştur
        session_id = secrets.token_urlsafe(16)
            
        return jsonify({
            'success': True,
            'token': token,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Mülakat token hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Token alınırken bir hata oluştu'
        })

@app.route('/realtime_chat', methods=['POST'])
async def realtime_chat():
    try:
        data = request.get_json()
        message = data.get('message')
        session_id = data.get('session_id')
        interview_code = data.get('interview_code')
        
        if not message or not session_id or not interview_code:
            return jsonify({
                'success': False,
                'error': 'Mesaj, session_id ve interview_code gerekli'
            }), 400
            
        # JSON dosyasından mülakat verilerini al
        json_path = os.path.join('interviews', f'{interview_code}.json')
        if not os.path.exists(json_path):
            return jsonify({
                'success': False,
                'error': 'Mülakat bilgileri bulunamadı'
            }), 404
            
        with open(json_path, 'r', encoding='utf-8') as f:
            interview_data = json.load(f)
            
        # Konuşma geçmişini al veya oluştur
        conversation_history = interview_data.get('conversation_history', [])
        current_question_index = interview_data.get('current_question_index', 0)
        
        # Sistem promptunu hazırla
        system_prompt = f"""Sen deneyimli ve empatik bir İK uzmanısın. Şu anda {interview_data.get('candidate_name')} ile {interview_data.get('position')} pozisyonu için mülakat yapıyorsun.

        Mülakat soruları:
        {json.dumps(interview_data.get('questions', []), indent=2, ensure_ascii=False)}

        Şu anki soru indeksi: {current_question_index}
        
        Konuşma geçmişi:
        {json.dumps(conversation_history, indent=2, ensure_ascii=False)}

        Önemli Kurallar:
        1. Adayın hazır olma durumunu mutlaka kontrol et:
           - "Hazır değilim" veya benzeri bir yanıt gelirse:
             * "Anlıyorum, acele etmeyelim. Hazır olduğunuzda başlayalım. Kendinizi rahat hissettiğinizde bana haber verebilirsiniz."
             * "Biraz daha zaman ister misiniz? Ben buradayım, hazır olduğunuzda başlayabiliriz."
           - Aday hazır olduğunu belirtene kadar bir sonraki soruya geçme
        
        2. Soru tekrarı isteklerini dikkatle dinle:
           - "Anlamadım", "Tekrar eder misiniz" gibi ifadelerde:
             * "Tabii ki, soruyu farklı bir şekilde açıklayayım..."
             * "Elbette, şöyle sorayım..."
           - Soruyu farklı kelimelerle, daha açıklayıcı bir şekilde tekrar et
           - Aday anlayana kadar bir sonraki soruya geçme
        
        3. Her yanıttan sonra:
           - Adayın cevabını anladığını göster
           - Kısa ve samimi bir yorum yap
           - Eğer cevap yetersizse, nazikçe detay iste
           - Aday hazırsa ve cevap tamamsa, bir sonraki konuya doğal bir geçiş yap
        
        4. Doğal konuşma örnekleri:
           - "Bu konudaki deneyimlerinizi dinlemek çok değerli. Peki, [sonraki konu] hakkında ne düşünüyorsunuz?"
           - "Anlıyorum, yaklaşımınız ilginç. İsterseniz şimdi biraz da [yeni konu] hakkında konuşalım..."
           - "Bu tecrübeleriniz etkileyici. Başka bir konuya geçmeden önce eklemek istediğiniz bir şey var mı?"
        
        5. Eğer aday gergin veya stresli görünüyorsa:
           - "Kendinizi rahat hissetmeniz çok önemli. Acele etmeyelim..."
           - "Bu sadece bir sohbet, kendinizi baskı altında hissetmeyin..."
           - "İsterseniz biraz ara verebiliriz..."
        
        6. Mülakat bitişi:
           - Tüm sorular tamamlandığında nazik bir kapanış yap
           - "Paylaştığınız değerli bilgiler için teşekkür ederim. Görüşmemizi burada sonlandıralım. Size en kısa sürede dönüş yapacağız. MÜLAKAT_BİTTİ"
        
        En önemli nokta: Bu bir robot-insan konuşması değil, iki insan arasında geçen doğal bir sohbet olmalı. Adayın her tepkisine ve ihtiyacına uygun şekilde yanıt ver. Acele etme, adayın rahat hissetmesini sağla."""

        # OpenAI API'ye istek gönder
        try:
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *conversation_history[-5:],  # Son 5 mesajı kullan
                        {"role": "user", "content": message}
                    ],
                    temperature=0.9,
                    max_tokens=250
                )
            )
            
            # Yanıtı al
            gpt_response = completion.choices[0].message.content
            is_interview_ended = "MÜLAKAT_BİTTİ" in gpt_response
            
            # Konuşma geçmişini güncelle
            conversation_history.append({"role": "user", "content": message})
            conversation_history.append({"role": "assistant", "content": gpt_response})
            
            # Soru indeksini güncelle (eğer bir soru sorulduysa)
            if "?" in gpt_response and not is_interview_ended:
                current_question_index += 1
            
            # JSON dosyasını güncelle
            interview_data['conversation_history'] = conversation_history
            interview_data['current_question_index'] = current_question_index
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(interview_data, f, ensure_ascii=False, indent=2)
            
            if is_interview_ended:
                try:
                    # Rapor oluşturma işlemini başlat
                    await generate_interview_report(interview_code, interview_data)
                    logger.info(f"Rapor oluşturma başarılı: {interview_code}")
                except Exception as e:
                    logger.error(f"Rapor oluşturma hatası: {str(e)}")
                # MÜLAKAT_BİTTİ kelimesini çıkar
                gpt_response = gpt_response.replace("MÜLAKAT_BİTTİ", "")
            
            return jsonify({
                'success': True,
                'text': gpt_response,
                'interview_ended': is_interview_ended
            })
            
        except Exception as e:
            logger.error(f"OpenAI API hatası: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'OpenAI API hatası: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Gerçek zamanlı sohbet hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

async def generate_interview_report(interview_code, interview_data):
    try:
        # JSON dosyasını kontrol et
        json_path = os.path.join('interviews', f'{interview_code}.json')
        if not os.path.exists(json_path):
            raise ValueError(f"Mülakat dosyası bulunamadı: {interview_code}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            interview_json = json.load(f)
            
        # Webhook URL'sini kontrol et
        webhook_rapor_url = interview_json.get('webhook_rapor_url', WEBHOOK_RAPOR_URL)
        
        # PDF oluştur
        pdf_path = os.path.join('reports', f'{interview_code}.pdf')
        
        # Değerlendirme verilerini hazırla
        evaluation_data = {
            "teknik_yetkinlik": interview_data.get('teknik_yetkinlik', 0),
            "iletisim_becerileri": interview_data.get('iletisim_becerileri', 0),
            "problem_cozme": interview_data.get('problem_cozme', 0),
            "genel_degerlendirme": interview_data.get('genel_degerlendirme', ''),
            "guclu_yonler": interview_data.get('guclu_yonler', []),
            "gelisim_alanlari": interview_data.get('gelisim_alanlari', [])
        }
        
        # Webhook'a gönderilecek veriyi hazırla
        webhook_data = {
            "mulakat_kodu": interview_code,
            "aday_bilgileri": {
                "isim": interview_data.get('candidate_name'),
                "pozisyon": interview_data.get('position'),
                "tarih": datetime.now().isoformat()
            },
            "degerlendirme": evaluation_data,
            "rapor_dosyasi": pdf_path
        }
        
        # Webhook'a gönder
        webhook_response = requests.post(
            webhook_rapor_url,
            json=webhook_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if webhook_response.status_code != 200:
            logger.error(f"Webhook hatası: {webhook_response.status_code} - {webhook_response.text}")
        
        # Mülakat durumunu güncelle
        interview_json['status'] = 'completed'
        interview_json['report_path'] = pdf_path
        interview_json['evaluation_data'] = evaluation_data
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(interview_json, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Mülakat raporu oluşturuldu ve webhook\'a gönderildi',
            'report_path': pdf_path
        })
        
    except Exception as e:
        logger.error(f"Rapor oluşturma hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/end_interview', methods=['POST'])
async def end_interview():
    try:
        data = request.get_json()
        interview_code = data.get('interview_code')
        conversation_history = data.get('conversation_history', [])
        
        if not interview_code:
            return jsonify({
                'success': False,
                'error': 'Mülakat kodu gerekli'
            }), 400
            
        # Rapor oluşturma işlemini başlat
        await generate_interview_report(interview_code, conversation_history)
        
        return jsonify({
            'success': True,
            'message': 'Mülakat sonlandırıldı ve rapor oluşturma işlemi başlatıldı'
        })
        
    except Exception as e:
        logger.error(f"Mülakat sonlandırma hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# window.onbeforeunload event'i için endpoint
@app.route('/save_interview_state', methods=['POST'])
def save_interview_state():
    try:
        data = request.json
        code = data.get('code')
        conversation_history = data.get('conversation_history', [])
        ended = data.get('ended', False)
        
        if not code:
            return jsonify({'success': False, 'error': 'Mülakat kodu gereklidir'}), 400
            
        # JSON dosyasını kontrol et
        json_path = os.path.join('interviews', f'{code}.json')
        if not os.path.exists(json_path):
            return jsonify({'success': False, 'error': 'Mülakat bulunamadı'}), 404
            
        # JSON dosyasını oku
        with open(json_path, 'r', encoding='utf-8') as f:
            interview_data = json.load(f)
            
        # Verileri güncelle
        interview_data['conversation_history'] = conversation_history
        if ended:
            interview_data['ended'] = True
            interview_data['status'] = 'completed'
            interview_data['ended_at'] = datetime.now().isoformat()
            
        # Verileri kaydet
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(interview_data, f, ensure_ascii=False, indent=2)
            
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Mülakat durumu kaydetme hatası: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_photo', methods=['POST'])
def save_photo():
    try:
        data = request.json
        code = data.get('code')
        photo_data = data.get('photo')
        timestamp = data.get('timestamp')
        
        if not code or not photo_data:
            return jsonify({'success': False, 'error': 'Mülakat kodu ve fotoğraf verisi gereklidir'}), 400
            
        # JSON dosyasını kontrol et
        json_path = os.path.join('interviews', f'{code}.json')
        if not os.path.exists(json_path):
            return jsonify({'success': False, 'error': 'Mülakat bulunamadı'}), 404
        
        # Base64 formatındaki fotoğrafı işle
        # Başlangıç kısmını (data:image/jpeg;base64,) kaldır
        image_data = photo_data.split(',')[1]
        
        # Fotoğrafları saklamak için klasör oluştur
        photos_dir = os.path.join('interviews', f'{code}_photos')
        os.makedirs(photos_dir, exist_ok=True)
        
        # Fotoğrafa benzersiz bir ad ver
        photo_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        photo_path = os.path.join(photos_dir, photo_filename)
        
        # Fotoğrafı kaydet
        with open(photo_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
            
        # JSON dosyasını güncelle (fotoğraf bilgilerini ekle)
        with open(json_path, 'r', encoding='utf-8') as f:
            interview_data = json.load(f)
            
        # Fotoğraf bilgilerini ekle
        if 'photos' not in interview_data:
            interview_data['photos'] = []
            
        interview_data['photos'].append({
            'filename': photo_filename,
            'timestamp': timestamp,
            'path': os.path.join(f'{code}_photos', photo_filename)
        })
        
        # Güncellenen veriyi kaydet
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(interview_data, f, ensure_ascii=False, indent=2)
            
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Fotoğraf kaydetme hatası: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_speech', methods=['GET'])
async def get_speech():
    try:
        text = request.args.get('text')
        if not text:
            return 'Text parameter is required', 400

        # Geçici ses dosyası için benzersiz bir isim oluştur
        temp_filename = f"temp/speech_{int(time.time())}_{random.randint(100000, 999999)}.mp3"
        
        try:
            # OpenAI TTS ile sesi oluştur - context manager kullanmadan
            tts_response = openai_client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            
            # Ses dosyasını kaydet
            tts_response.stream_to_file(temp_filename)
            
            # Dosyayı oku ve yanıt olarak gönder
            def generate():
                with open(temp_filename, 'rb') as f:
                    while chunk := f.read(8192):
                        yield chunk
                # Dosyayı sil
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            
            return Response(
                generate(),
                mimetype='audio/mpeg',
                headers={
                    'Content-Disposition': 'inline',
                    'Cache-Control': 'no-cache'
                }
            )
            
        except Exception as e:
            logger.error(f"TTS hatası: {str(e)}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return jsonify({
                'success': False,
                'error': 'Ses oluşturulamadı'
            }), 500
            
    except Exception as e:
        logger.error(f"Ses endpoint hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate_report', methods=['POST'])
async def generate_report():
    try:
        logging.info("Rapor oluşturma süreci başlatıldı...")
        
        # İstek verilerini al
        data = request.get_json()
        interview_code = data.get('interview_code')
        conversation_history = data.get('conversation_history', [])
        
        if not interview_code:
            return jsonify({'success': False, 'error': 'Mülakat kodu gerekli'}), 400
            
        # Mülakat dosyasını kontrol et
        interview_file = f'interviews/{interview_code}.json'
        if not os.path.exists(interview_file):
            return jsonify({'success': False, 'error': 'Mülakat bulunamadı'}), 404
            
        # Mülakat verilerini oku
        with open(interview_file, 'r', encoding='utf-8') as f:
            interview_data = json.load(f)
            
        if not conversation_history:
            return jsonify({'success': False, 'error': 'Konuşma geçmişi boş'}), 400
            
        # GPT değerlendirmesi için prompt hazırla
        prompt = f"""
        Aşağıdaki mülakat için detaylı bir değerlendirme raporu hazırla:
        
        Aday: {interview_data.get('candidate_name')}
        Pozisyon: {interview_data.get('position')}
        Tarih: {interview_data.get('created_at')}
        
        Mülakat Konuşma Akışı:
        """
        
        # Konuşma akışını ekle
        for message in conversation_history:
            role = "Aday" if message['role'] == 'user' else "Mülakat Asistanı"
            prompt += f"\n{role}: {message['content']}\n"
            
        prompt += """
        Lütfen aşağıdaki kriterlere göre değerlendirme yap:
        1. Teknik Yetkinlik (1-10)
        2. İletişim Becerileri (1-10)
        3. Problem Çözme (1-10)
        4. Deneyim Seviyesi (1-10)
        
        Her kriter için detaylı açıklama ve örnekler ver.
        Güçlü yönler ve gelişim alanlarını belirt.
        Genel değerlendirme ve tavsiyeler ekle.
        """
        
        # GPT'den değerlendirme al
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": "Sen bir mülakat değerlendirme uzmanısın."},
                     {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        evaluation_text = response.choices[0].message.content
        
        # Skorları ve bölümleri çıkar
        scores = {
            'technical': extract_score(evaluation_text, 'Teknik Yetkinlik'),
            'communication': extract_score(evaluation_text, 'İletişim Becerileri'),
            'problem_solving': extract_score(evaluation_text, 'Problem Çözme'),
            'experience': extract_score(evaluation_text, 'Deneyim Seviyesi')
        }
        
        sections = {
            'strengths': extract_section(evaluation_text, 'Güçlü Yönler'),
            'improvements': extract_section(evaluation_text, 'Gelişim Alanları'),
            'overall': extract_section(evaluation_text, 'Genel Değerlendirme')
        }
        
        # Genel skoru hesapla
        overall_score = calculate_overall_score(scores)
        
        # PDF rapor dosya adını oluştur - mülakat kodu ile isimlendirme
        pdf_filename = f"{interview_code}_report.pdf"
        pdf_path = os.path.join('reports', pdf_filename)
        
        # PDF raporu oluştur
        os.makedirs('reports', exist_ok=True)  # reports klasörünün varlığından emin ol
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Başlık
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph(f"Mülakat Değerlendirme Raporu - {interview_data.get('candidate_name')}", title_style))
        
        # Temel Bilgiler
        info_style = ParagraphStyle(
            'Info',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20
        )
        elements.append(Paragraph(f"Pozisyon: {interview_data.get('position')}", info_style))
        elements.append(Paragraph(f"Tarih: {interview_data.get('created_at')}", info_style))
        elements.append(Paragraph(f"Genel Skor: {overall_score}/10", info_style))
        
        # Skorlar
        elements.append(Paragraph("Değerlendirme Skorları", styles['Heading2']))
        for criterion, score in scores.items():
            elements.append(Paragraph(f"{criterion.replace('_', ' ').title()}: {score}/10", info_style))
        
        # Bölümler
        for section_title, content in sections.items():
            elements.append(Paragraph(section_title.replace('_', ' ').title(), styles['Heading2']))
            elements.append(Paragraph(content, info_style))
        
        # PDF'i oluştur
        doc.build(elements)
        
        # PDF içeriğini base64 ile kodla
        with open(pdf_path, "rb") as pdf_file:
            pdf_content_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
        
        # Rapor URL'sini oluştur
        report_url = url_for('serve_report', filename=pdf_filename, _external=True)
        
        # Rapor bilgilerini JSON dosyasına ekle
        interview_data['report'] = {
            'url': report_url,
            'pdf_base64': pdf_content_base64,
            'created_at': datetime.now().isoformat(),
            'filename': pdf_filename
        }
        
        # Güncellenmiş mülakat verilerini kaydet
        with open(interview_file, 'w', encoding='utf-8') as f:
            json.dump(interview_data, f, ensure_ascii=False, indent=2)
        
        # Webhook'a gönder
        try:
            send_report_webhook(pdf_path, evaluation_text, pdf_content_base64)
        except Exception as e:
            logging.error(f"Webhook gönderme hatası: {str(e)}")
            # Webhook hatası olsa bile işlem devam edecek
        
        return jsonify({
            'success': True,
            'report_url': report_url,
            'pdf_included': True
        })
        
    except Exception as e:
        logging.error(f"Rapor oluşturma hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def send_report_webhook(pdf_path, evaluation_data, pdf_content_base64=None):
    """Raporu webhook'a gönder"""
    try:
        # PDF dosya adını URL'ye dönüştür
        pdf_filename = os.path.basename(pdf_path)
        report_url = url_for('serve_report', filename=pdf_filename, _external=True)
        
        # PDF içeriğini base64 ile kodla (eğer parametre olarak gelmemişse)
        if pdf_content_base64 is None:
            with open(pdf_path, "rb") as pdf_file:
                pdf_content_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
        
        # Mülakat verilerini hazırla
        interview_summary = {
            "mulakat_kodu": current_interview.code,
            "aday_bilgileri": {
                "isim": current_interview.candidate_name,
                "pozisyon": current_interview.position,
                "cv_ozeti": current_interview.cv_summary,
                "on_bilgi": current_interview.pre_info,
                "gereksinimler": current_interview.requirements,
                "tarih": current_interview.start_time.isoformat()
            },
            "mulakat_metrikleri": {
                "iletisim_puani": current_interview.metrics["iletisim_puani"],
                "ozguven_puani": current_interview.metrics["ozguven_puani"],
                "teknik_bilgi": current_interview.metrics["teknik_bilgi"],
                "genel_puan": current_interview.metrics["genel_puan"]
            },
            "degerlendirme": evaluation_data,
            "konusma_akisi": current_interview.conversation_history,
            "rapor_url": report_url,  # PDF URL'si
            "rapor_pdf_base64": pdf_content_base64,  # PDF içeriği base64 olarak
            "olusturulma_tarihi": datetime.now().isoformat()
        }
        
        # JSON dosyasından webhook URL'sini kontrol et
        json_path = os.path.join('interviews', f'{current_interview.code}.json')
        webhook_url = WEBHOOK_RAPOR_URL  # Varsayılan URL
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                interview_json = json.load(f)
                if 'webhook_rapor_url' in interview_json:
                    webhook_url = interview_json['webhook_rapor_url']
                    logger.info(f"JSON'dan webhook URL'si kullanılıyor: {webhook_url}")
                else:
                    logger.info(f"Varsayılan webhook URL'si kullanılıyor: {webhook_url}")
        
        # WebhookRapor'a gönder
        response = requests.post(
            webhook_url,
            json=interview_summary,
            headers={
                'Content-Type': 'application/json',
                'X-Interview-Code': current_interview.code
            }
        )
        
        if response.status_code == 200:
            logger.info(f"Rapor webhook'a başarıyla gönderildi: {current_interview.code}")
            return True
        else:
            logger.error(f"Rapor webhook'a gönderilemedi: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Webhook gönderme hatası: {str(e)}")
        return False

if __name__ == '__main__':
    try:
        print("\n=== AIVA Mülakat Sistemi Başlatılıyor ===")
        
        # Dosya izleme sistemini başlat
        observer = start_file_watcher()
        
        # Flask uygulamasını başlat
        print("\n[*] Web sunucusu başlatılıyor (Port: 5004)...")
        app.run(host='0.0.0.0', port=5004)
        
    except Exception as e:
        print(f"\n[!] Program başlatılamadı: {str(e)}")
    finally:
        if 'observer' in locals() and observer:
            observer.stop()
            observer.join()
            