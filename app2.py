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
from flask import Flask, request, jsonify, render_template, send_from_directory
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
import secrets
import string

# Flask ve async ayarları
app = Flask(__name__)
CORS(app)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Logger ayarları
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
# Doğru yol
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(__file__), 'google_credentials.json')

# reports klasörünü oluştur
if not os.path.exists('reports'):
    os.makedirs('reports')

# temp klasörü oluştur
if not os.path.exists('temp'):
    os.makedirs('temp')

# E-posta ayarları
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
REPORT_SENDER = os.getenv('REPORT_SENDER')
REPORT_RECIPIENT = os.getenv('REPORT_RECIPIENT')

# Global değişkenler
interview_data = {}  # Mülakat verilerini saklamak için
interview_codes = {}  # Mülakat kodlarını saklamak için

def generate_interview_code():
    """Benzersiz bir mülakat kodu oluştur"""
    code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    return code

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
        self.requirements = []
        self.custom_questions = []
        self.conversation_history = []
        self.sentiment_scores = []
        self.metrics = {
            "iletisim_puani": 0,
            "ozguven_puani": 0,
            "teknik_bilgi": 0,
            "genel_puan": 0
        }
        self.start_time = datetime.now()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.current_question_index = 0
        self.interview_questions = []
        
        self.reports_dir = "reports"
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)

    def set_interview_details(self, code):
        """Mülakat detaylarını ayarla"""
        if code not in interview_data:
            raise ValueError("Geçersiz mülakat kodu")

        interview = interview_data[code]
        self.candidate_name = interview["candidate_info"]["name"]
        self.position = interview["candidate_info"]["position"]
        self.requirements = interview["candidate_info"]["requirements"]
        self.custom_questions = interview["candidate_info"]["custom_questions"]
        
        # Soruları hazırla
        self._prepare_interview_questions()

    def _prepare_interview_questions(self):
        """Özelleştirilmiş mülakat sorularını hazırla"""
        try:
            # Özel soruları kullan
            self.interview_questions = self.custom_questions

            # İlk soruyu ekle
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Merhaba {self.candidate_name}, {self.position} pozisyonu için mülakatımıza hoş geldiniz. " + self.interview_questions[0]
            })
            self.current_question_index = 1

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
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Merhaba {self.candidate_name}, {self.position} pozisyonu için mülakatımıza hoş geldiniz. " + self.interview_questions[0]
            })
            self.current_question_index = 1

    async def get_gpt_response(self, text):
        """Mülakat bağlamında GPT yanıtını al ve bir sonraki soruyu hazırla"""
        try:
            if not text:
                logger.warning("Boş metin için GPT yanıtı istenemez")
                return None
                
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
            next_question = ""
            if self.current_question_index < len(self.interview_questions):
                next_question = "\n\nBir sonraki sorum: " + self.interview_questions[self.current_question_index]
                self.current_question_index += 1
            else:
                next_question = "\n\nMülakat sona erdi. Katılımınız için teşekkür ederiz."
            
            # Konuşma geçmişini güncelle
            self.conversation_history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": evaluation + next_question}
            ])
            
            return evaluation + next_question
            
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
        """Mülakat yanıtını işle ve tüm analizleri yap"""
        try:
            # GPT yanıtını al
            response = await self.get_gpt_response(text)
            if not response:
                return None
                
            # Duygu analizi yap
            analysis = await self._analyze_sentiment(text)
            
            # Sonuçları logla
            logger.info(f"""
            Mülakat İlerlemesi:
            Aday: {self.candidate_name}
            Son Yanıt: {text[:50]}...
            GPT Değerlendirmesi: {response[:50]}...
            Analiz: {analysis}
            Güncel Puanlar: {self.metrics}
            """)
            
            return response
            
        except Exception as e:
            logger.error(f"Mülakat yanıt işleme hatası: {str(e)}")
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

            logger.info("Rapor e-posta ile gönderildi")
            return True
        except Exception as e:
            logger.error(f"E-posta gönderme hatası: {str(e)}")
            return False

    def send_report_webhook(self, report_data):
        try:
            webhook_url = 'https://otomasyon.aivatech.io/api/v1/webhooks/B7iYtwVltWEzX2nvAaWCX'
            
            logger.info(f"Webhook'a gönderilen veri: {json.dumps(report_data, indent=2)}")
            
            response = requests.post(
                webhook_url,
                json=report_data,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Webhook hatası: {response.status_code} - {response.text}")
                return False
                
            logger.info("Rapor webhook'a başarıyla gönderildi")
            return True
            
        except Exception as e:
            logger.error(f"Webhook gönderme hatası: {str(e)}")
            return False

# Global değişken
current_interview = None

@app.route('/')
def index():
    return render_template('index.html')


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
        
        # Interview sınıfını başlat
        current_interview = InterviewAssistant()  # parametresiz başlat
        current_interview.set_interview_details(  # detayları sonradan set et
            data['candidate_name'],
            data['position']
        )
        
        logger.info(f"Mülakat başlatıldı: {data['candidate_name']} - {data['position']}")
        
        return jsonify({
            "success": True,
            "message": "Mülakat başarıyla başlatıldı"
        })
        
    except Exception as e:
        logger.error(f"Mülakat başlatma hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Mülakat başlatılamadı: {str(e)}"
        }), 500

@app.route('/process_audio', methods=['POST'])
async def process_audio():
    try:
        if 'audio' not in request.files:
            logger.error("Ses dosyası bulunamadı")
            return jsonify({"success": False, "error": "Ses dosyası bulunamadı"}), 400
            
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
            
            # FFmpeg komutu - basitleştirilmiş ve daha güvenilir
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-i', webm_path,
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                temp_path
            ]
            
            subprocess.run(ffmpeg_command, check=True, capture_output=True)
            
            # Google Speech client
            client = speech.SpeechClient()
            
            with open(temp_path, 'rb') as audio_file:
                content = audio_file.read()
            
            # Ses tanıma yapılandırması - daha hassas ayarlar
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="tr-TR",
                enable_automatic_punctuation=True,
                use_enhanced=True,
                audio_channel_count=1,
                enable_word_time_offsets=True,
                enable_word_confidence=True,
                profanity_filter=False
            )
            
            # Ses tanıma işlemi
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.recognize(config=config, audio=audio)
            )
            
            if not response.results:
                logger.warning("Ses tanınamadı - Sonuç boş")
                return jsonify({
                    "success": False,
                    "error": "Ses tanınamadı, lütfen daha yüksek sesle ve net konuşun",
                    "continue_listening": True
                }), 400
                
            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            
            logger.info(f"Tanınan metin: {transcript} (Güven: {confidence:.2f})")
            
            if confidence < 0.6:
                return jsonify({
                    "success": False,
                    "error": "Ses net anlaşılamadı, lütfen daha yüksek sesle ve net konuşun",
                    "continue_listening": True
                }), 400
            
            # GPT yanıtı al
            if current_interview:
                # GPT yanıtını hemen al
                gpt_response = await current_interview.get_gpt_response(transcript)
                
                # Analizi arka planda yap
                asyncio.create_task(current_interview._analyze_sentiment(transcript))
                
                return jsonify({
                    "success": True,
                    "transcript": transcript,
                    "response": gpt_response,
                    "metrics": current_interview.metrics,
                    "continue_listening": True
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
        logger.error(f"Genel hata: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "continue_listening": True
        }), 500
   
@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        global current_interview
        if not current_interview:
            return jsonify({
                "success": False,
                "error": "Aktif mülakat bulunamadı"
            }), 400

        logger.info("Mülakat sonlandırma işlemi başlatıldı")

        # Rapor oluştur
        pdf_path = current_interview.generate_pdf_report()
        if not pdf_path:
            logger.error("PDF raporu oluşturulamadı")
            return jsonify({
                "success": False,
                "error": "Rapor oluşturulamadı"
            }), 500

        logger.info(f"PDF raporu oluşturuldu: {pdf_path}")

        # Rapor URL'i oluştur
        report_filename = os.path.basename(pdf_path)
        report_url = f"/reports/{report_filename}"

        # Webhook için rapor verilerini hazırla
        report_data = {
            "aday_bilgileri": {
                "isim": current_interview.candidate_name,
                "pozisyon": current_interview.position,
                "tarih": current_interview.start_time.strftime('%d.%m.%Y %H:%M'),
                "mulakat_kodu": request.args.get('code', '')
            },
            "metrikler": current_interview.metrics,
            "konusma_akisi": [
                {
                    "rol": entry["role"],
                    "mesaj": entry["content"]
                } for entry in current_interview.conversation_history
            ],
            "rapor_url": request.host_url.rstrip('/') + report_url
        }

        # E-posta göndermeyi dene
        try:
            email_sent = current_interview.send_report_email(pdf_path)
            if email_sent:
                logger.info("Rapor e-posta ile gönderildi")
            else:
                logger.warning("Rapor e-posta ile gönderilemedi")
        except Exception as e:
            logger.error(f"E-posta gönderme hatası: {str(e)}")

        # Webhook'a gönder
        try:
            webhook_sent = current_interview.send_report_webhook(report_data)
            if webhook_sent:
                logger.info("Rapor webhook'a gönderildi")
            else:
                logger.warning("Rapor webhook'a gönderilemedi")
        except Exception as e:
            logger.error(f"Webhook gönderme hatası: {str(e)}")

        # Mülakat nesnesini sıfırla
        current_interview = None

        logger.info("Mülakat sonlandırma işlemi tamamlandı")
        
        return jsonify({
            "success": True,
            "message": "Mülakat başarıyla sonlandırıldı",
            "report_url": report_url,
            "report_data": report_data
        })

    except Exception as e:
        logger.error(f"Mülakat durdurma hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Mülakat sonlandırılamadı: {str(e)}"
        }), 500

# Rapor dosyalarını sunmak için endpoint
@app.route('/reports/<path:filename>')
def serve_report(filename):
    try:
        logger.info(f"Rapor dosyası istendi: {filename}")
        reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
        return send_from_directory(reports_dir, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Rapor sunma hatası: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Rapor bulunamadı"
        }), 404

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

@app.route('/webhook/interview', methods=['POST'])
def receive_interview_data():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Veri bulunamadı"}), 400

        # Mülakat kodu oluştur
        interview_code = generate_interview_code()
        
        # Mülakat verilerini sakla
        interview_data[interview_code] = {
            "candidate_info": {
                "name": data.get("adSoyad"),
                "email": data.get("mail"),
                "position": data.get("isIlaniPozisyonu"),
                "requirements": data.get("isIlaniGereksinimleri", []),
                "custom_questions": data.get("mulakatSorulari", [])
            },
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "status": "pending"
        }

        # Webhook yanıtını hazırla
        response_data = {
            "success": True,
            "interview_code": interview_code,
            "interview_url": f"{request.host_url}interview?code={interview_code}",
            "expires_at": interview_data[interview_code]["expires_at"]
        }

        # Webhook'a mülakat kodunu gönder
        try:
            webhook_response = requests.post(
                'https://otomasyon.aivatech.io/api/v1/webhooks/B7iYtwVltWEzX2nvAaWCX',
                json=response_data,
                headers={'Content-Type': 'application/json'}
            )
            if not webhook_response.ok:
                logger.error(f"Webhook yanıt hatası: {webhook_response.text}")
        except Exception as e:
            logger.error(f"Webhook gönderme hatası: {str(e)}")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Webhook alım hatası: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/interview')
def interview_page():
    code = request.args.get('code')
    if not code or code not in interview_data:
        return "Geçersiz veya süresi dolmuş mülakat kodu", 404

    interview = interview_data[code]
    if datetime.fromisoformat(interview["expires_at"]) < datetime.now():
        return "Mülakat süresi dolmuş", 410

    return render_template('interview.html', 
                         interview={
                             "candidate_name": interview["candidate_info"]["name"],
                             "position": interview["candidate_info"]["position"],
                             "code": code,
                             "created_at": interview["created_at"]
                         })

if __name__ == '__main__':
    try:
        asgi_app = WsgiToAsgi(app)
        app.run(host='0.0.0.0', port=5004)
    except Exception as e:
        logger.critical(f"Program başlatılamadı: {str(e)}")
