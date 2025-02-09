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
from flask import Flask, request, jsonify, render_template, redirect, send_from_directory, Response
from flask_cors import CORS
from datetime import datetime, timedelta
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
from gtts import gTTS

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
openai_client = OpenAI()

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

        self.eleven_api_key = os.getenv("ELEVEN_API_KEY")
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel sesi - değiştirilebilir

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
        """
        Metni Eleven Labs API kullanarak sese çevirir
        """
        try:
            if not text:
                return
                
            # GPT konuşma durumunu güncelle
            interview_data['is_gpt_speaking'] = True
                
            # Eleven Labs API endpoint
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.eleven_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.75,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data, headers=headers) as response:
                        if response.status == 200:
                            # Ses akışını al
                            audio_stream = await response.read()
                            
                            # Geçici ses dosyasını kaydet
                            temp_file = "temp_speech.mp3"
                            with open(temp_file, "wb") as f:
                                f.write(audio_stream)
                            
                            # Ses dosyasını çal
                            playsound.playsound(temp_file)
                            
                            # Geçici dosyayı sil
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        else:
                            logger.error(f"Eleven Labs API Hatası: {response.status}")
                            await self._fallback_tts(text)
            except Exception as e:
                logger.error(f"Eleven Labs bağlantı hatası: {str(e)}")
                await self._fallback_tts(text)
                        
        except Exception as e:
            logger.error(f"Ses üretme hatası: {str(e)}")
            await self._fallback_tts(text)
        finally:
            # GPT konuşma durumunu güncelle
            interview_data['is_gpt_speaking'] = False
            
    async def _fallback_tts(self, text):
        """
        Yedek TTS sistemi (gTTS kullanarak)
        """
        try:
            logger.info("Yedek TTS sistemi kullanılıyor...")
            tts = gTTS(text=text, lang='tr')
            temp_file = "temp_speech.mp3"
            tts.save(temp_file)
            playsound.playsound(temp_file)
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            logger.error(f"Yedek TTS hatası: {str(e)}")

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

            # Hoşgeldin mesajını hazırla
            welcome_message = f"""Merhaba {self.candidate_name}, {self.position} pozisyonu için mülakatımıza hoş geldiniz. 
            
            Özgeçmişinizde belirttiğiniz {self.cv_summary} bilgilerine dayanarak, pozisyonun gerektirdiği {', '.join(self.requirements)} yetkinlikleri hakkında konuşacağız.
            
            {self.pre_info if self.pre_info else ''}
            
            Başlamak için hazır mısınız?"""

            self.conversation_history.append({
                "role": "assistant",
                "content": welcome_message
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
                "Zorlu bir iş durumunu nasıl çözdüğünüzü anlatır mısınız?",
                "Gelecekteki kariyer hedefleriniz nelerdir?"
            ]
            welcome_message = f"Merhaba {self.candidate_name}, {self.position} pozisyonu için mülakatımıza hoş geldiniz."
            self.conversation_history.append({
                "role": "assistant",
                "content": welcome_message
            })

    async def get_gpt_response(self, text):
        """Mülakat bağlamında GPT yanıtını al ve bir sonraki soruyu hazırla"""
        try:
            if not text:
                logger.warning("Boş metin için GPT yanıtı istenemez")
                return None

            # Konuşma geçmişini güncelle
            self.conversation_history.append({"role": "user", "content": text})
            
            # Cevabı değerlendir ve metrikleri güncelle
            await self._analyze_sentiment(text)
            
            # Mülakat tamamlandı mı kontrol et
            if self.current_question_index >= len(self.interview_questions):
                # Mülakat bitti, son mesajı gönder
                final_message = """Mülakat sona erdi. Katılımınız için teşekkür ederiz. 
                Değerlendirme raporunuz hazırlanıyor..."""
                
                self.conversation_history.append({"role": "assistant", "content": final_message})
                
                # PDF raporu oluştur
                pdf_path = self.generate_pdf_report()
                
                # Değerlendirme verilerini hazırla
                evaluation_data = {
                    "teknik_yetkinlik": self.metrics["teknik_bilgi"],
                    "iletisim_becerileri": self.metrics["iletisim_puani"],
                    "ozguven": self.metrics["ozguven_puani"],
                    "genel_puan": self.metrics["genel_puan"],
                    "guclu_yonler": [],
                    "gelistirilmesi_gereken_yonler": [],
                    "genel_degerlendirme": ""
                }
                
                if pdf_path:
                    # Raporu webhook'a gönder
                    await self.send_report_webhook(pdf_path, evaluation_data)
                    
                return final_message, True  # True -> Mülakat bitti
            
            # Bir sonraki soruyu hazırla
            next_question = self.interview_questions[self.current_question_index]
            self.current_question_index += 1
            
            # Soruyu kaydet ve gönder
            self.conversation_history.append({"role": "assistant", "content": next_question})
            
            return next_question, False  # False -> Mülakat devam ediyor
            
        except Exception as e:
            logger.error(f"GPT yanıt hatası: {str(e)}")
            return None

    async def _analyze_sentiment(self, text):
        """Metni analiz et ve metrikleri güncelle"""
        try:
            # Analiz promptu
            sentiment_prompt = f"""
            Lütfen aşağıdaki cevabı analiz et ve şu metrikleri 0-100 arası puanla:
            
            Pozisyon: {self.position}
            Soru: {self.interview_questions[self.current_question_index-1]}
            Cevap: "{text}"
            
            Şu formatta JSON yanıt ver:
            {{
                "iletisim_becerisi": [puan],
                "ozguven": [puan],
                "teknik_bilgi": [puan],
                "yorum": "[kısa değerlendirme]"
            }}
            """
            
            response = await openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen bir kıdemli yazılım mühendisi ve mülakat uzmanısın."},
                    {"role": "user", "content": sentiment_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            # JSON yanıtı parse et
            analysis = json.loads(response.choices[0].message.content)
            
            # Metrikleri güncelle
            self.sentiment_scores.append(analysis)
            
            # Ortalama puanları hesapla
            total_scores = len(self.sentiment_scores)
            self.metrics = {
                "iletisim_puani": sum(s["iletisim_becerisi"] for s in self.sentiment_scores) / total_scores,
                "ozguven_puani": sum(s["ozguven"] for s in self.sentiment_scores) / total_scores,
                "teknik_bilgi": sum(s["teknik_bilgi"] for s in self.sentiment_scores) / total_scores
            }
            self.metrics["genel_puan"] = sum(self.metrics.values()) / 3
            
            logger.debug(f"Metrikler güncellendi: {self.metrics}")
            
        except Exception as e:
            logger.error(f"Duygu analizi hatası: {str(e)}")

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
                "rapor_dosyasi": pdf_path,
                "olusturulma_tarihi": datetime.now().isoformat()
            }
            
            # WebhookRapor'a gönder
            response = requests.post(
                WEBHOOK_RAPOR_URL,
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
                logger.error(f"Webhook hatası: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Webhook gönderme hatası: {str(e)}")
            return False

def generate_interview_code():
    """Benzersiz bir mülakat kodu oluştur"""
    code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    return code

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
        return "Mülakat kodu gereklidir", 400
        
    try:
        # Önce memory'de kontrol et
        if code not in interview_data:
            # JSON dosyasından okuma yapalım
            json_path = os.path.join('interviews', f'{code}.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    interview_json = json.load(f)
                    
                # Interview data'yı güncelle
                interview_data[code] = {
                    "candidate_info": {
                        "name": interview_json.get("candidate_name"),
                        "position": interview_json.get("position"),
                        "requirements": interview_json.get("requirements", []),
                        "custom_questions": interview_json.get("questions", [])
                    },
                    "created_at": interview_json.get("created_at", datetime.now().isoformat()),
                    "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
                    "status": "active"
                }
            else:
                return "Geçersiz veya süresi dolmuş mülakat kodu", 404

        interview = interview_data[code]
        
        # Global current_interview'ı oluştur
        global current_interview
        current_interview = InterviewAssistant()
        current_interview.set_interview_details(code)
        
        return render_template('interview.html', 
                             interview={
                                 "candidate_name": interview["candidate_info"]["name"],
                                 "position": interview["candidate_info"]["position"],
                                 "code": code,
                                 "created_at": interview["created_at"]
                             })
                             
    except Exception as e:
        logger.error(f"Mülakat sayfası yükleme hatası: {str(e)}")
        return "Mülakat yüklenirken bir hata oluştu", 500

@app.route('/create_interview', methods=['POST'])
def create_interview():
    """Yeni bir mülakat oluştur"""
    try:
        data = request.get_json()
        candidate_name = data.get('candidate_name')
        position = data.get('position')
        
        if not candidate_name or not position:
            return jsonify({
                'success': False,
                'error': 'Aday adı ve pozisyon gereklidir'
            })
            
        try:
            # GPT ile pozisyona özel sorular oluştur
            prompt = f"""
            {position} pozisyonu için ön görüşme soruları oluştur.
            
            Aday Bilgileri:
            - İsim: {candidate_name}
            - Pozisyon: {position}
            
            Lütfen aşağıdaki kriterlere göre 3-4 arası kısa ve öz soru oluştur:
            1. Genel deneyim ve beklentiler
            2. Pozisyona ilgi ve motivasyon
            3. Temel teknik bilgi seviyesi
            
            Önemli Notlar:
            - Bu bir ön görüşmedir, detaylı teknik sorular sormayın
            - Her soru kısa ve öz olmalı
            - Adayın genel profilini anlamaya odaklanın
            - Cevapları çok detaylı değerlendirmeyin, bir sonraki soruya geçin
            
            Soruları liste formatında ver ve her soru Türkçe olsun.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen bir kıdemli yazılım mühendisi ve mülakat uzmanısın."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # GPT yanıtından soruları al
            custom_questions = response.choices[0].message.content.strip().split('\n')
            # Boş satırları temizle
            custom_questions = [q.strip() for q in custom_questions if q.strip()]
            
            logger.info(f"GPT tarafından {len(custom_questions)} soru oluşturuldu")
            
        except Exception as e:
            logger.error(f"GPT soru oluşturma hatası: {str(e)}")
            # Hata durumunda varsayılan soruları kullan
            custom_questions = [
                "1. Yapay zekanın temel bileşenleri hakkında bilgi verebilir misiniz?",
                "2. Belirli bir veri setinde overfitting problemini nasıl tanımlar ve çözersiniz?",
                "3. Günlük çalışmalarınızda hangi yapay zeka frameworklerini kullandınız?",
                "4. Çeşitli regresyon teknikleri hakkında bilgi verebilir misiniz?",
                "5. NLP konusunda ne gibi deneyimleriniz var?"
            ]
            
        # Benzersiz kod oluştur
        interview_code = generate_interview_code()
        
        # Mülakat verilerini hazırla
        interview_info = {
            "candidate_info": {
                "name": candidate_name,
                "position": position,
                "requirements": data.get('requirements', []),
                "custom_questions": custom_questions
            },
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "status": "active"
        }
        
        try:
            # Memory'ye kaydet
            interview_data[interview_code] = interview_info
            
            # JSON dosyasına kaydet
            json_path = os.path.join('interviews', f'{interview_code}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "code": interview_code,
                    "candidate_name": candidate_name,
                    "position": position,
                    "requirements": interview_info["candidate_info"]["requirements"],
                    "questions": custom_questions,
                    "created_at": interview_info["created_at"],
                    "status": "active"
                }, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Yeni mülakat oluşturuldu: {interview_code}")
            
            # Mülakat linkini oluştur
            domain = os.getenv('DOMAIN_NAME', request.host_url.rstrip('/'))
            
            # HTTPS kontrolü
            protocol = 'https' if request.is_secure or 'https' in request.host_url else 'http'
            
            # Ana domain ve alt domainler için farklı URL'ler oluştur
            interview_urls = {
                'main': f"{protocol}://{domain}/interview?code={interview_code}",
                'create': f"{protocol}://{domain}/",
                'join': f"{protocol}://{domain}/join"
            }
            
            return jsonify({
                'success': True,
                'code': interview_code,
                'urls': interview_urls,
                'questions': custom_questions
            })
            
        except Exception as e:
            logger.error(f"Mülakat kaydetme hatası: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Mülakat kaydedilemedi'
            })
            
    except Exception as e:
        logger.error(f"Mülakat oluşturma hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/verify_code', methods=['POST'])
def verify_code():
    """Mülakat kodunu doğrula"""
    try:
        data = request.get_json()
        code = data.get('code')

        if not code:
            return jsonify({
                'success': False,
                'error': 'Mülakat kodu gereklidir'
            })

        try:
            # Önce memory'de kontrol et
            if code in interview_data:
                return jsonify({'success': True})
                
            # JSON dosyasından kontrol et
            json_path = os.path.join('interviews', f'{code}.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    interview_json = json.load(f)
                    
                # Interview data'yı güncelle
                interview_data[code] = {
                    "candidate_info": {
                        "name": interview_json.get("candidate_name"),
                        "position": interview_json.get("position"),
                        "requirements": interview_json.get("requirements", []),
                        "custom_questions": interview_json.get("questions", [])
                    },
                    "created_at": interview_json.get("created_at", datetime.now().isoformat()),
                    "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
                    "status": "active"
                }
                
                return jsonify({'success': True})
                
            return jsonify({
                'success': False,
                'error': 'Geçersiz mülakat kodu'
            })
            
        except Exception as e:
            logger.error(f"Kod doğrulama işlem hatası: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Kod doğrulanamadı'
            })
            
    except Exception as e:
        logger.error(f"Kod doğrulama genel hatası: {str(e)}")
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
        # GPT konuşuyorsa ses kaydını işleme
        if interview_data.get('is_gpt_speaking', False):
            return jsonify({
                'success': False,
                'error': 'GPT konuşurken ses kaydı alınamaz',
                'continue_listening': True
            })
            
        interview_code = request.form.get('interview_code')
        if not interview_code:
            return jsonify({
                'success': False,
                'error': 'Mülakat kodu gerekli'
            })
            
        # Ses dosyasını al
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({
                'success': False,
                'error': 'Ses dosyası bulunamadı'
            })
            
        # JSON dosyasından mülakat verilerini al
        json_path = os.path.join('interviews', f'{interview_code}.json')
        if not os.path.exists(json_path):
            return jsonify({
                "success": False,
                "error": "Mülakat bilgileri bulunamadı",
                "continue_listening": True
            }), 404
            
        with open(json_path, 'r', encoding='utf-8') as f:
            interview_data = json.load(f)

        # Geçici dizini kontrol et ve oluştur
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
            
        # Benzersiz dosya isimleri oluştur
        timestamp = str(time.time()).replace('.', '_')
        temp_webm = os.path.join(temp_dir, f'audio_{timestamp}.webm')
        temp_wav = os.path.join(temp_dir, f'audio_{timestamp}.wav')
        
        temp_files = [temp_webm, temp_wav]
        
        try:
            # WebM dosyasını kaydet
            audio_file.save(temp_webm)
            logger.info(f"WebM dosyası kaydedildi: {temp_webm}")
            
            if not os.path.exists(temp_webm):
                raise FileNotFoundError(f"WebM dosyası oluşturulamadı: {temp_webm}")
            
            # FFmpeg ile WAV'a dönüştür
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-i', temp_webm,
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                temp_wav
            ]
            
            # FFmpeg işlemini çalıştır
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"FFmpeg hatası: {stderr.decode()}")
            
            if not os.path.exists(temp_wav):
                raise FileNotFoundError(f"WAV dosyası oluşturulamadı: {temp_wav}")
            
            # Ses seviyesi kontrolü
            data, _ = sf.read(temp_wav)
            volume_level = float(np.max(np.abs(data))) * 100
            
            if volume_level < SILENCE_THRESHOLD:
                return jsonify({
                    "success": False,
                    "error": "Ses seviyesi çok düşük, lütfen daha yüksek sesle konuşun",
                    "continue_listening": True,
                    "should_restart": True
                })
            
            # Google Speech client
            client = speech.SpeechClient()
            
            with open(temp_wav, 'rb') as audio_file:
                content = audio_file.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="tr-TR",
                enable_automatic_punctuation=True,
                use_enhanced=True,
                audio_channel_count=1,
                enable_separate_recognition_per_channel=False,
                speech_contexts=[{
                    "phrases": ["merhaba", "evet", "hayır", "tamam", "anladım"],
                    "boost": 20.0
                }],
                profanity_filter=False,  # Sansürlemeyi kapat
                enable_word_time_offsets=False  # Zaman damgalarına gerek yok
            )
            
            # Ses tanıma işlemi
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.recognize(config=config, audio=audio)
            )
            
            if not response.results:
                return jsonify({
                    "success": False,
                    "error": "Ses tanınamadı, lütfen tekrar konuşun",
                    "continue_listening": True,
                    "should_restart": True
                })
            
            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence

            # Realtime chat endpoint'ine yönlendir
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{request.host_url}realtime_chat",
                    json={
                        "message": transcript,
                        "session_id": "audio_session",
                        "interview_code": interview_code
                    }
                ) as chat_response:
                    chat_data = await chat_response.json()
                    
                    if not chat_data.get('success'):
                        return jsonify({
                            "success": False,
                            "error": chat_data.get('error', 'GPT yanıt hatası'),
                            "continue_listening": True
                        }), 500
                    
                    return jsonify({
                        "success": True,
                        "transcript": transcript,
                        "response": chat_data.get("text", ""),
                        "confidence": confidence,
                        "interview_ended": chat_data.get("interview_ended", False)
                    })
            
        except FileNotFoundError as e:
            logger.error(f"Dosya bulunamadı hatası: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Dosya bulunamadı: {str(e)}",
                "continue_listening": True
            }), 500
            
        except Exception as e:
            logger.error(f"Ses işleme hatası: {str(e)}")
            return jsonify({
                "success": False,
                "error": str(e),
                "continue_listening": True
            }), 500
            
    except Exception as e:
        logger.error(f"Genel hata: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "continue_listening": True
        }), 500
        
    finally:
        # Geçici dosyaları temizle
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Geçici dosya silindi: {file_path}")
            except Exception as e:
                logger.error(f"Dosya silme hatası ({file_path}): {str(e)}")

@app.route('/generate_report', methods=['POST'])
async def generate_report():
    try:
        logger.info("Rapor oluşturma süreci başlatıldı...")
        
        if not current_interview:
            error_msg = "Mülakat oturumu bulunamadı"
            logger.error(error_msg)
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400

        # Konuşma geçmişini al
        conversation_history = request.json.get('conversation_history', [])
        if not conversation_history:
            error_msg = "Konuşma geçmişi boş"
            logger.error(error_msg)
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400
            
        logger.info("Konuşma geçmişi alındı, GPT değerlendirmesi başlıyor...")
        
        try:
            # GPT ile değerlendirme yap
            logger.info("GPT değerlendirmesi başlatılıyor...")
            evaluation_prompt = f"""
            Aşağıdaki mülakat konuşmasını değerlendir ve bir rapor oluştur:
            
            Aday: {current_interview.candidate_name if current_interview else "Bilinmiyor"}
            Pozisyon: {current_interview.position if current_interview else "Bilinmiyor"}
            
            İş İlanı Gereksinimleri:
            {json.dumps(current_interview.requirements if current_interview else [], indent=2, ensure_ascii=False)}
            
            Konuşma Geçmişi:
            {json.dumps(conversation_history, indent=2, ensure_ascii=False)}
            
            Lütfen aşağıdaki başlıklara göre değerlendirme yap:
            1. Teknik Bilgi ve Deneyim (100 üzerinden puan ver)
            2. İletişim Becerileri (100 üzerinden puan ver)
            3. Problem Çözme Yeteneği (100 üzerinden puan ver)
            4. Pozisyon Gereksinimlerine Uygunluk (100 üzerinden puan ver)
            5. Güçlü Yönler
            6. Geliştirilmesi Gereken Yönler
            7. Genel Değerlendirme ve Tavsiye
            
            Yanıtı JSON formatında ver.
            """
            
            evaluation_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Sen bir kıdemli yazılım mühendisi ve mülakat uzmanısın."},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
            )
            
            evaluation = evaluation_response.choices[0].message.content
            evaluation_data = json.loads(evaluation)
            logger.info("GPT değerlendirmesi tamamlandı")
            
            try:
                # PDF oluştur
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = os.path.join('reports', f"mulakat_raporu_{timestamp}.pdf")
                
                # PDF oluşturma işlemleri...
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
                story.append(Paragraph("Mülakat Değerlendirme Raporu", title_style))
                story.append(Spacer(1, 12))
                
                # Aday Bilgileri
                info_style = ParagraphStyle(
                    'Info',
                    parent=styles['Normal'],
                    fontSize=12,
                    spaceAfter=6
                )
                story.append(Paragraph(f"Aday: {current_interview.candidate_name if current_interview else 'Bilinmiyor'}", info_style))
                story.append(Paragraph(f"Pozisyon: {current_interview.position if current_interview else 'Bilinmiyor'}", info_style))
                story.append(Paragraph(f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}", info_style))
                story.append(Spacer(1, 20))
                
                # Değerlendirme Puanları
                story.append(Paragraph("Değerlendirme Puanları", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                metrics_data = [
                    ["Değerlendirme Kriteri", "Puan"],
                    ["Teknik Bilgi ve Deneyim", f"{evaluation_data.get('teknik_bilgi', 0)}/100"],
                    ["İletişim Becerileri", f"{evaluation_data.get('iletisim_becerileri', 0)}/100"],
                    ["Problem Çözme Yeteneği", f"{evaluation_data.get('problem_cozme', 0)}/100"],
                    ["Pozisyon Uygunluğu", f"{evaluation_data.get('pozisyon_uygunlugu', 0)}/100"]
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
                
                # Detaylı Değerlendirme
                story.append(Paragraph("Detaylı Değerlendirme", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                story.append(Paragraph("<b>Güçlü Yönler:</b>", styles['Normal']))
                story.append(Paragraph(evaluation_data.get('guclu_yonler', ''), styles['Normal']))
                story.append(Spacer(1, 12))
                
                story.append(Paragraph("<b>Geliştirilmesi Gereken Yönler:</b>", styles['Normal']))
                story.append(Paragraph(evaluation_data.get('gelistirilmesi_gereken_yonler', ''), styles['Normal']))
                story.append(Spacer(1, 12))
                
                story.append(Paragraph("<b>Genel Değerlendirme ve Tavsiye:</b>", styles['Normal']))
                story.append(Paragraph(evaluation_data.get('genel_degerlendirme', ''), styles['Normal']))
                story.append(Spacer(1, 20))
                
                # Konuşma Geçmişi
                story.append(Paragraph("Mülakat Detayları", styles['Heading2']))
                story.append(Spacer(1, 12))
                
                for entry in conversation_history:
                    if entry["role"] == "user":
                        story.append(Paragraph(f"<b>Aday:</b> {entry['content']}", styles['Normal']))
                    else:
                        story.append(Paragraph(f"<b>Mülakat Uzmanı:</b> {entry['content']}", styles['Normal']))
                    story.append(Spacer(1, 12))
                
                # PDF oluştur
                doc.build(story)
                logger.info(f"PDF raporu başarıyla oluşturuldu: {pdf_path}")
                
                try:
                    # Webhook'a gönder
                    webhook_data = {
                        "mulakat_kodu": current_interview.code if current_interview else "UNKNOWN",
                        "aday_bilgileri": {
                            "isim": current_interview.candidate_name if current_interview else "Bilinmiyor",
                            "pozisyon": current_interview.position if current_interview else "Bilinmiyor",
                            "tarih": datetime.now().isoformat()
                        },
                        "degerlendirme": evaluation_data,
                        "rapor_dosyasi": pdf_path
                    }
                    
                    webhook_response = requests.post(
                        WEBHOOK_RAPOR_URL,
                        json=webhook_data,
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if webhook_response.status_code == 200:
                        logger.info("Rapor webhook'a başarıyla gönderildi")
                    else:
                        logger.error(f"Webhook hatası: {webhook_response.status_code} - {webhook_response.text}")
                    
                    return jsonify({
                        "success": True,
                        "message": "Rapor başarıyla oluşturuldu ve gönderildi",
                        "pdf_path": pdf_path
                    })
                    
                except Exception as e:
                    error_msg = f"Webhook gönderme hatası: {str(e)}"
                    logger.error(error_msg)
                    return jsonify({
                        "success": False,
                        "error": error_msg
                    }), 500
                    
            except Exception as e:
                error_msg = f"PDF oluşturma hatası: {str(e)}"
                logger.error(error_msg)
                return jsonify({
                    "success": False,
                    "error": error_msg
                }), 500
                
        except Exception as e:
            error_msg = f"GPT değerlendirme hatası: {str(e)}"
            logger.error(error_msg)
            return jsonify({
                "success": False,
                "error": error_msg
            }), 500
            
    except Exception as e:
        error_msg = f"Rapor oluşturma genel hatası: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "success": False,
            "error": error_msg
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
        conn.execute(
            'INSERT INTO interview_codes (email, code, created_at) VALUES (?, ?, ?)',
            (email, code, datetime.now().isoformat())
        )
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
def webhook_aday_handler():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Veri bulunamadı"}), 400

        # Mülakat kodu oluştur
        interview_code = generate_interview_code()
        
        # JSON dosyasını oluştur
        interview_data = {
            "code": interview_code,
            "candidate_name": data.get("adSoyad"),
            "position": data.get("isIlaniPozisyonu"),
            "requirements": data.get("isIlaniGereksinimleri", []),
            "questions": data.get("mulakatSorulari", []),
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # JSON dosyasını kaydet
        json_path = os.path.join('interviews', f'{interview_code}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(interview_data, f, ensure_ascii=False, indent=2)
        
        # WebhookAdayİşlem'e yanıt gönder
        response_data = {
            "success": True,
            "interview_code": interview_code,
            "interview_url": f"{request.host_url}interview?code={interview_code}",
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        # Webhook'a bildirim gönder
        try:
            webhook_response = requests.post(
                WEBHOOK_ADAY_URL,
                json=response_data,
                headers={'Content-Type': 'application/json'}
            )
            logger.info(f"WebhookAdayİşlem yanıtı: {webhook_response.status_code}")
        except Exception as e:
            logger.error(f"WebhookAdayİşlem gönderme hatası: {str(e)}")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Webhook alım hatası: {str(e)}")
        return jsonify({"error": str(e)}), 500

def send_report_webhook(pdf_path, evaluation_data):
    """Raporu webhook'a gönder"""
    try:
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
            "rapor_dosyasi": pdf_path,
            "olusturulma_tarihi": datetime.now().isoformat()
        }
        
        # WebhookRapor'a gönder
        response = requests.post(
            WEBHOOK_RAPOR_URL,
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
            logger.error(f"Webhook hatası: {response.status_code} - {response.text}")
            return False
                
    except Exception as e:
        logger.error(f"Webhook gönderme hatası: {str(e)}")
        return False

@app.route('/reports/<path:filename>')
def serve_report(filename):
    """Rapor dosyalarını sun"""
    try:
        return send_from_directory('reports', filename)
    except Exception as e:
        logger.error(f"Rapor sunma hatası: {str(e)}")
        return "Rapor bulunamadı", 404
    
    
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
        system_prompt = f"""Sen deneyimli ve samimi bir İK uzmanısın. Şu anda {interview_data.get('candidate_name')} ile {interview_data.get('position')} pozisyonu için mülakat yapıyorsun.

        İşte sana verilen sorular:
        {json.dumps(interview_data.get('questions', []), indent=2, ensure_ascii=False)}

        Şu anki soru indeksi: {current_question_index}
        
        Konuşma geçmişi:
        {json.dumps(conversation_history, indent=2, ensure_ascii=False)}

        Nasıl davranmalısın:
        1. Çok samimi ve doğal bir şekilde konuş, sanki karşında gerçekten biri varmış gibi
        2. Her cevaptan sonra o cevapla ilgili kısa bir yorum yap. Örneğin:
           - "Ah, bu konudaki deneyiminizi duymak güzel..."
           - "İlginç bir yaklaşım, teşekkür ederim..."
           - "Bu zorluklarla nasıl başa çıktığınızı anlamak önemli..."
        3. Eğer adayın cevabını anlamazsan, nazikçe tekrar sorup açıklama iste:
           - "Affedersiniz, tam anlayamadım. Biraz daha açıklayabilir misiniz?"
           - "Bu konuyu biraz daha detaylandırabilir misiniz?"
        4. Bir sonraki soruya geçerken doğal geçişler kullan:
           - "Anlıyorum, çok teşekkürler. Şimdi başka bir konuya geçelim..."
           - "Bu deneyimleriniz gerçekten değerli. İsterseniz şimdi..."
           - "Güzel, bu konuyu anladım. Peki, size şunu sormak istiyorum..."
        5. Eğer aday "merhaba" derse, sıcak bir karşılama yap:
           - "Merhaba [isim] Bey/Hanım, bugün sizinle tanışmak çok güzel. Nasılsınız?"
        6. Mülakat bittiğinde nazik bir kapanış yap:
           - "Değerli vaktiniz ve paylaşımlarınız için çok teşekkür ederim. Görüşmemizi burada sonlandıralım. Size en kısa sürede dönüş yapacağız. MÜLAKAT_BİTTİ"

        Önemli: Sadece verilen soruları sor ama bunu çok doğal bir sohbet havasında yap. Her cevaptan sonra mutlaka kısa bir yorum yap ve sonra diğer soruya geç."""
            
        # OpenAI API'ye istek gönder
        try:
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    temperature=0.7,
                    max_tokens=150
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
        # Rapor oluşturma promptu
        report_prompt = f"""Aşağıdaki mülakat konuşmasını analiz et ve bir değerlendirme raporu hazırla:

        Pozisyon: {interview_data.get('position')}
        Aday: {interview_data.get('candidate_name')}
        
        Mülakat Konuşması:
        {json.dumps(interview_data.get('conversation_history', []), indent=2, ensure_ascii=False)}
        
        Lütfen aşağıdaki başlıklara göre bir rapor hazırla:
        1. Teknik Yetkinlik
        2. İletişim Becerileri
        3. Problem Çözme Yaklaşımı
        4. Güçlü Yönler
        5. Gelişime Açık Alanlar
        6. Genel Değerlendirme ve Öneri"""

        # OpenAI API'ye istek gönder
        completion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen bir İK uzmanısın. Mülakat değerlendirme raporu hazırlayacaksın."},
                    {"role": "user", "content": report_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
        )
        
        # Raporu al
        report = completion.choices[0].message.content
        
        # Raporu kaydet
        report_path = os.path.join('reports', f'{interview_code}.txt')
        os.makedirs('reports', exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        # Mülakat durumunu güncelle
        interview_data['status'] = 'completed'
        interview_data['report_path'] = report_path
        
        json_path = os.path.join('interviews', f'{interview_code}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(interview_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Mülakat raporu oluşturuldu: {report_path}")
        
    except Exception as e:
        logger.error(f"Rapor oluşturma hatası: {str(e)}")
        raise

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
async def save_interview_state():
    """Mülakat sayfası kapatıldığında çağrılır"""
    try:
        data = request.get_json()
        interview_code = data.get('interview_code')
        
        if not interview_code:
            return jsonify({
                'success': False,
                'error': 'Mülakat kodu gerekli'
            }), 400
            
        # JSON dosyasını oku
        json_path = os.path.join('interviews', f'{interview_code}.json')
        if not os.path.exists(json_path):
            return jsonify({
                'success': False,
                'error': 'Mülakat dosyası bulunamadı'
            }), 404
            
        with open(json_path, 'r', encoding='utf-8') as f:
            interview_data = json.load(f)
        
        # PDF raporu oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join('reports', f"mulakat_raporu_{timestamp}.pdf")
        
        # PDF oluşturma işlemleri
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
        story.append(Paragraph("Mülakat Değerlendirme Raporu", title_style))
        story.append(Spacer(1, 12))
        
        # Aday Bilgileri
        info_style = ParagraphStyle(
            'Info',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=6
        )
        story.append(Paragraph(f"Aday: {interview_data.get('candidate_name')}", info_style))
        story.append(Paragraph(f"Pozisyon: {interview_data.get('position')}", info_style))
        story.append(Paragraph(f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}", info_style))
        story.append(Spacer(1, 20))
        
        # GPT ile değerlendirme yap
        evaluation_prompt = f"""
        Lütfen aşağıdaki mülakat verilerini analiz ederek detaylı bir değerlendirme raporu hazırla:
        
        Aday: {interview_data.get('candidate_name')}
        Pozisyon: {interview_data.get('position')}
        
        Mülakat Konuşması:
        {json.dumps(interview_data.get('conversation_history', []), indent=2, ensure_ascii=False)}
        
        Lütfen şu başlıklara göre değerlendirme yap:
        1. Teknik Yetkinlik (100 üzerinden puan)
        2. İletişim Becerileri (100 üzerinden puan)
        3. Problem Çözme Yaklaşımı (100 üzerinden puan)
        4. Güçlü Yönler
        5. Geliştirilmesi Gereken Alanlar
        6. Genel Değerlendirme ve İşe Uygunluk
        
        Yanıtı JSON formatında ver.
        """
        
        completion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen bir kıdemli İK uzmanısın."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
        )
        
        evaluation_data = json.loads(completion.choices[0].message.content)
        
        # Değerlendirme Puanları
        story.append(Paragraph("Değerlendirme Puanları", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        metrics_data = [
            ["Değerlendirme Kriteri", "Puan"],
            ["Teknik Yetkinlik", f"{evaluation_data.get('teknik_yetkinlik', 0)}/100"],
            ["İletişim Becerileri", f"{evaluation_data.get('iletisim_becerileri', 0)}/100"],
            ["Problem Çözme", f"{evaluation_data.get('problem_cozme', 0)}/100"]
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
        
        # Detaylı Değerlendirme
        story.append(Paragraph("Detaylı Değerlendirme", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("<b>Güçlü Yönler:</b>", styles['Normal']))
        story.append(Paragraph(evaluation_data.get('guclu_yonler', ''), styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("<b>Geliştirilmesi Gereken Alanlar:</b>", styles['Normal']))
        story.append(Paragraph(evaluation_data.get('gelistirilmesi_gereken_alanlar', ''), styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("<b>Genel Değerlendirme:</b>", styles['Normal']))
        story.append(Paragraph(evaluation_data.get('genel_degerlendirme', ''), styles['Normal']))
        
        # PDF oluştur
        doc.build(story)
        
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
            WEBHOOK_RAPOR_URL,
            json=webhook_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if webhook_response.status_code != 200:
            logger.error(f"Webhook hatası: {webhook_response.status_code} - {webhook_response.text}")
        
        # Mülakat durumunu güncelle
        interview_data['status'] = 'completed'
        interview_data['report_path'] = pdf_path
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(interview_data, f, ensure_ascii=False, indent=2)
        
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
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    
    
        
        
            



    
    
    