<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIVA Mülakat Asistanı</title>
    <link href="https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Ana Tema Renkleri */
        :root {
            --primary: #fbbf24;
            --primary-light: #fde68a;
            --primary-dark: #f59e0b;
            --background: #fffbeb;
            --text: #1f2937;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(135deg, var(--background) 0%, #fff7e6 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            margin: 0;
            padding: 0;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            padding: 1rem;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 50;
        }

        .chat-container {
            width: 100%;
            max-width: 1000px;
            height: calc(100vh - 80px);
            margin: 80px auto 0;
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* Logo Yeni Pozisyon */
        .aiva-logo {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 150px;
            height: auto;
            z-index: 100;
            filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
            transition: transform 0.3s ease;
        }

        .aiva-logo:hover {
            transform: scale(1.05);
        }

        /* Modern Avatar */
        .aiva-avatar {
            width: 450px;
            height: 450px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 4rem auto;
            box-shadow: 
                0 20px 40px rgba(251, 191, 36, 0.3),
                inset 0 -10px 20px rgba(0, 0, 0, 0.1);
            position: relative;
            cursor: pointer;
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            overflow: hidden;
        }

        .aiva-avatar::before {
            content: '';
            position: absolute;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.2) 0%, transparent 60%);
            animation: rotateGradient 8s linear infinite;
        }

        @keyframes rotateGradient {
            0% { transform: translate(-50%, -50%) rotate(0deg); }
            100% { transform: translate(-50%, -50%) rotate(360deg); }
        }

        /* Dinleme Animasyonu */
        .listening-animation {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            pointer-events: none;
        }

        .listening-animation::before,
        .listening-animation::after {
            content: '';
            position: absolute;
            inset: -20px;
            border: 3px solid var(--primary-light);
            border-radius: 50%;
            animation: pulseWave 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            opacity: 0;
        }

        .listening-animation::after {
            animation-delay: 1s;
        }

        @keyframes pulseWave {
            0% { transform: scale(0.8); opacity: 0.8; }
            100% { transform: scale(1.2); opacity: 0; }
        }

        /* Konuşma Animasyonu */
        .speaking {
            animation: speakingAnimation 4s ease-in-out infinite;
        }

        @keyframes speakingAnimation {
            0% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(10px, -15px) rotate(2deg); }
            50% { transform: translate(-5px, 10px) rotate(-1deg); }
            75% { transform: translate(-10px, -5px) rotate(1deg); }
            100% { transform: translate(0, 0) rotate(0deg); }
        }

        /* Mod Göstergesi */
        .mode-indicator {
            position: absolute;
            bottom: -60px;
            left: 50%;
            transform: translateX(-50%);
            background: white;
            padding: 0.8rem 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1rem;
            color: var(--text);
            transition: all 0.3s ease;
        }

        .mode-indicator .icon {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--primary);
            transition: background-color 0.3s ease;
        }

        .mode-indicator.listening .icon {
            animation: blink 1s infinite;
        }

        /* Chat Baloncukları */
        .chat-bubble {
            max-width: 80%;
            padding: 1.5rem 2rem;
            border-radius: 1.5rem;
            position: relative;
            animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            line-height: 1.6;
            background: white;
            border: 1px solid rgba(251, 191, 36, 0.2);
        }

        .assistant-bubble {
            margin-right: auto;
            border-bottom-left-radius: 0.5rem;
            background: linear-gradient(135deg, white 0%, #fffbeb 100%);
        }

        .user-bubble {
            margin-left: auto;
            border-bottom-right-radius: 0.5rem;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
        }

        .assistant-bubble::before {
            content: '';
            position: absolute;
            bottom: -0.5rem;
            left: 1rem;
            background: white;
            border-right: 1px solid #e5e7eb;
            border-bottom: 1px solid #e5e7eb;
            clip-path: polygon(0 0, 0 100%, 100% 0);
            width: 1rem;
            height: 1rem;
        }

        .user-bubble::before {
            content: '';
            position: absolute;
            bottom: -0.5rem;
            right: 1rem;
            background: #4f46e5;
            clip-path: polygon(0 0, 100% 100%, 100% 0);
            width: 1rem;
            height: 1rem;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            25% { transform: translateY(-15px) rotate(2deg); }
            50% { transform: translateY(0px) rotate(0deg); }
            75% { transform: translateY(15px) rotate(-2deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .pulse-ring {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            animation: pulse-ring 2s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
            border: 6px solid #4f46e5;
        }

        @keyframes pulse-ring {
            0% {
                transform: scale(0.8);
                opacity: 0.5;
            }
            50% {
                transform: scale(1.2);
                opacity: 0.2;
            }
            100% {
                transform: scale(0.8);
                opacity: 0.5;
            }
        }

        .status-indicator {
            position: absolute;
            bottom: -1rem;
            left: 50%;
            transform: translateX(-50%);
            background: white;
            padding: 0.5rem 1rem;
            border-radius: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: #4f46e5;
            transition: all 0.3s ease;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #6b7280;
            transition: background-color 0.3s ease;
        }

        .status-dot.listening {
            background: #ef4444;
            animation: blink 1s infinite;
        }

        .auto-mode .status-indicator {
            background: #4f46e5;
            color: white;
        }

        .auto-mode-indicator {
            display: inline-flex;
            align-items: center;
            margin-left: 0.5rem;
            animation: pulse 2s infinite;
        }

        .volume-waves {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            pointer-events: none;
        }

        .volume-waves::before,
        .volume-waves::after {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: rgba(79, 70, 229, 0.2);
            animation: waves 2s infinite;
        }

        .volume-waves::after {
            animation-delay: 1s;
        }

        @keyframes waves {
            0% {
                transform: scale(1);
                opacity: 0.5;
            }
            100% {
                transform: scale(1.5);
                opacity: 0;
            }
        }

        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .completed {
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        }
        
        .paused {
            opacity: 0.7;
        }

        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(5px);
        }

        .modal-content {
            background: white;
            padding: 2.5rem;
            border-radius: 1.5rem;
            max-width: 600px;
            width: 90%;
            position: relative;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(251, 191, 36, 0.2);
        }

        .modal-header {
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
            color: var(--primary-dark);
            text-align: center;
        }

        .modal-body {
            margin-bottom: 2rem;
        }

        .modal-body p {
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }

        .modal-body ul {
            margin-left: 1.5rem;
            margin-top: 0.5rem;
        }

        .modal-body li {
            margin-bottom: 0.5rem;
            color: var(--text);
        }

        .modal-footer {
            text-align: center;
        }

        .modal-button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 1rem 2rem;
            border-radius: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
            font-size: 1.1rem;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(251, 191, 36, 0.3);
        }

        .modal-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(251, 191, 36, 0.4);
        }
    </style>
</head>
<body>
    <!-- AIVA Logo -->
    <img src="https://www.aivatech.io/wp-content/uploads/2023/09/AIVA-App-Logo1-1200-x-300piksel-1-1-1024x256.png" alt="AIVA Logo" class="aiva-logo">

    <!-- Bilgilendirme Modal -->
    <div id="infoModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                Mülakat Bilgilendirmesi
            </div>
            <div class="modal-body">
                <p>�� <strong>Mülakat Başlatma:</strong>
                    <ul>
                        <li>Robot simgesine tıklayarak mülakatı başlatabilirsiniz</li>
                        <li>Mikrofon izni vermeniz gerekecektir</li>
                    </ul>
                </p>
                
                <p>🎤 <strong>Manuel Mod:</strong>
                    <ul>
                        <li>Space tuşuna basılı tutarak konuşun</li>
                        <li>Tuşu bıraktığınızda kayıt otomatik durur</li>
                    </ul>
                </p>
                
                <p>🔄 <strong>Otomatik Mod:</strong>
                    <ul>
                        <li>Robot simgesine ikinci kez tıklayarak otomatik moda geçebilirsiniz</li>
                        <li>Ses seviyeniz algılandığında kayıt otomatik başlar ve durur</li>
                    </ul>
                </p>
                
                <p>⚠️ <strong>Önemli Notlar:</strong>
                    <ul>
                        <li>GPT'nin konuşması bitene kadar bekleyin</li>
                        <li>"Otomatik dinleme aktif" yazısını görmeden konuşmaya başlamayın</li>
                        <li>Net ve yavaş konuşun</li>
                        <li>Cevaplarınızı çok geciktirmeyin</li>
                    </ul>
                </p>
            </div>
            <div class="modal-footer">
                <button class="modal-button" onclick="closeModal()">Anladım, Başla</button>
            </div>
        </div>
    </div>

        <!-- Header -->
    <div class="header">
        <div class="max-w-7xl mx-auto px-4">
            <div class="flex justify-between items-center">
                <div>
                    <h1 class="text-2xl font-bold text-gray-800">{{ interview.candidate_name }}</h1>
                    <p class="text-gray-600">{{ interview.position }}</p>
                </div>
                <div class="text-right">
                    <p class="text-sm text-gray-500">Mülakat Kodu: {{ interview.code }}</p>
                    <p class="text-xs text-gray-500">{{ interview.created_at }}</p>
                    </div>
                </div>
            </div>
        </div>

    <div class="chat-container">
        <!-- AIVA Avatar -->
        <div class="aiva-avatar" id="aivaAvatar">
            <div class="listening-animation"></div>
            <i class="fas fa-robot text-white text-7xl"></i>
            <div class="mode-indicator">
                <div class="icon" id="modeIcon"></div>
                <span id="modeText">Dinlemeye hazır</span>
                </div>
            </div>

        <!-- Chat Messages -->
        <div id="messages" class="chat-messages">
            <div class="chat-bubble assistant-bubble">
                <p>Merhaba {{ interview.candidate_name }}, {{ interview.position }} pozisyonu için mülakatımıza hoş geldiniz. Size nasıl yardımcı olabilirim?</p>
            </div>
        </div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let audioContext = null;
        let analyser = null;
        let silenceStart = null;
        let silenceTimeout = null;
        let isAutoMode = true;
        let isGPTSpeaking = false; // GPT konuşma durumu
        let interviewEnded = false; // Mülakat bitiş durumu
        let speechQueue = [];
        let isSpeaking = false;
        let utteranceQueue = [];
        let isProcessingUtterance = false;
        let audioStream = null;
        let isAudioInitialized = false;
        let isMediaRecorderReady = false;
        let interviewCode = new URLSearchParams(window.location.search).get('code');

        // Ses seviyesi eşikleri
        const SILENCE_THRESHOLD = 0.05;  // Sessizlik eşiği (app2.py'dan alındı)
        const VOICE_THRESHOLD = 0.15;    // Konuşma başlangıç eşiği
        const SILENCE_DURATION = 1500;   // Sessizlik süresi (ms)
        const MIN_CONFIDENCE = 0.6;      // Minimum güven skoru

        async function initAudio() {
            try {
                // Önceki ses akışını temizle
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                }
                
                // Önceki AudioContext'i kapat
                if (audioContext) {
                    await audioContext.close();
                }
                
                // Yeni ses akışı al
                audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        channelCount: 1,
                        sampleRate: 16000
                    }
                });
                
                // Yeni AudioContext oluştur
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(audioStream);
                source.connect(analyser);
                analyser.fftSize = 2048;
                
                // MediaRecorder'ı yeniden oluştur
                if (mediaRecorder) {
                    mediaRecorder.ondataavailable = null;
                    mediaRecorder.onstop = null;
                }
                
                mediaRecorder = new MediaRecorder(audioStream, {
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: 16000
                });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (!isGPTSpeaking && event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = async () => {
                    if (!isGPTSpeaking && audioChunks.length > 0) {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        if (audioBlob.size > 0) {
                            try {
                                await sendAudioToServer(audioBlob);
                            } catch (error) {
                                console.error('Ses gönderme hatası:', error);
                                showError('Ses işlenirken bir hata oluştu');
                            }
                        }
                        audioChunks = [];
                    }
                };
                
                isMediaRecorderReady = true;
                isAudioInitialized = true;
                return true;
            } catch (error) {
                console.error('Ses sistemi başlatma hatası:', error);
                showError('Mikrofon erişimi sağlanamadı');
                isMediaRecorderReady = false;
                isAudioInitialized = false;
                return false;
            }
        }

        async function stopAudioSystem() {
            try {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                }
                
                if (audioContext) {
                    await audioContext.suspend();
                }
                
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                }
                
                isAudioInitialized = false;
            } catch (error) {
                console.error('Ses sistemi durdurma hatası:', error);
            }
        }

        function startVoiceDetection() {
            if (!analyser || !isAutoMode || isSpeaking || isGPTSpeaking || isProcessingUtterance) return;
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
        function checkAudioLevel() {
                if (!isAutoMode || isSpeaking || isGPTSpeaking || isProcessingUtterance) return;

            analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((a, b) => a + b) / bufferLength;
            const volume = average / 128.0;
            
                updateVolumeIndicator(volume);

                if (!isRecording && volume > VOICE_THRESHOLD) {
                    startRecording();
                } else if (isRecording && volume < SILENCE_THRESHOLD) {
                if (!silenceStart) {
                    silenceStart = Date.now();
                    } else if (Date.now() - silenceStart > SILENCE_DURATION) {
                    stopRecording();
                        silenceStart = null;
                }
            } else {
                silenceStart = null;
            }
            
                if (isAutoMode && !isSpeaking && !isGPTSpeaking && !isProcessingUtterance) {
                requestAnimationFrame(checkAudioLevel);
            }
        }

            requestAnimationFrame(checkAudioLevel);
        }

        function updateVolumeIndicator(volume) {
            const volumeWaves = document.querySelector('.volume-waves');
            if (volumeWaves) {
                volumeWaves.style.opacity = volume;
            }
            
            const statusDot = document.getElementById('statusDot');
            if (volume > VOICE_THRESHOLD) {
                statusDot.style.backgroundColor = '#ef4444'; // Kırmızı
            } else if (volume > SILENCE_THRESHOLD) {
                statusDot.style.backgroundColor = '#22c55e'; // Yeşil
            } else {
                statusDot.style.backgroundColor = '#6b7280'; // Gri
            }
        }

        function toggleRecording() {
            if (!mediaRecorder) {
                initAudio().then((initialized) => {
                    if (initialized) {
                        isAutoMode = true;
                        startVoiceDetection();
                        updateStatus('Otomatik mod aktif', 'auto');
                    }
                });
                return;
            }

            if (isRecording) {
                stopRecording();
            } else {
                isAutoMode = true;
                startVoiceDetection();
                updateStatus('Otomatik mod aktif', 'auto');
            }
        }

        function startRecording() {
            if (!mediaRecorder || isRecording) return;
            
            try {
                if (mediaRecorder.state === 'recording') {
                    console.log('MediaRecorder zaten kayıt yapıyor');
                    return;
                }
                
                if (!isMediaRecorderReady) {
                    console.log('MediaRecorder hazır değil, yeniden başlatılıyor');
                    initAudio().then((initialized) => {
                        if (initialized) {
                mediaRecorder.start();
                            isRecording = true;
                            updateRecordingUI(true);
                        }
                    });
                    return;
                }
                
                mediaRecorder.start();
                isRecording = true;
                updateRecordingUI(true);
                
            } catch (error) {
                console.error('Kayıt başlatma hatası:', error);
                showError('Kayıt başlatılamadı');
                isRecording = false;
                isMediaRecorderReady = false;
                updateRecordingUI(false);
            }
        }

        function stopRecording() {
            if (!mediaRecorder || !isRecording) return;
            
            try {
                if (mediaRecorder.state === 'inactive') {
                    console.log('MediaRecorder zaten durmuş');
                    return;
                }
                
                mediaRecorder.stop();
                isRecording = false;
                updateRecordingUI(false);
                
            } catch (error) {
                console.error('Kayıt durdurma hatası:', error);
                showError('Kayıt durdurulamadı');
                isRecording = false;
                updateRecordingUI(false);
            }
        }

        function updateRecordingUI(isRecording) {
            const modeIcon = document.getElementById('modeIcon');
            const modeText = document.getElementById('modeText');
            const avatar = document.getElementById('aivaAvatar');
            const listeningAnimation = avatar.querySelector('.listening-animation');
            
            if (isRecording) {
                modeIcon.style.backgroundColor = '#ef4444';
                modeText.textContent = 'Kaydediliyor...';
                avatar.classList.add('speaking');
                listeningAnimation.style.display = 'block';
            } else {
                modeIcon.style.backgroundColor = isAutoMode ? 'var(--primary)' : '#6b7280';
                modeText.textContent = isAutoMode ? 'Dinleniyor...' : 'Dinlemeye hazır';
                avatar.classList.remove('speaking');
                listeningAnimation.style.display = isAutoMode ? 'block' : 'none';
            }
        }

        function updateStatus(text, mode) {
            const modeText = document.getElementById('modeText');
            const modeIcon = document.getElementById('modeIcon');
            const avatar = document.getElementById('aivaAvatar');
            
            modeText.textContent = text;
            
            if (mode === 'auto') {
                avatar.classList.add('auto-mode');
                modeIcon.style.backgroundColor = 'var(--primary)';
                avatar.querySelector('.listening-animation').style.display = 'block';
            } else if (mode === 'manual') {
                avatar.classList.remove('auto-mode');
                modeIcon.style.backgroundColor = '#6b7280';
                avatar.querySelector('.listening-animation').style.display = 'none';
            } else if (mode === 'recording') {
                modeIcon.style.backgroundColor = '#ef4444';
                avatar.classList.add('speaking');
            } else if (mode === 'speaking') {
                modeIcon.style.backgroundColor = 'var(--primary-dark)';
                avatar.classList.add('speaking');
            } else if (mode === 'paused') {
                modeIcon.style.backgroundColor = '#6b7280';
                avatar.classList.remove('speaking');
                avatar.querySelector('.listening-animation').style.display = 'none';
            }
        }

        async function speakResponse(text) {
            return new Promise((resolve) => {
                // Önceki konuşmaları temizle
                window.speechSynthesis.cancel();
                
                // Ses sistemini durdur
                stopAudioSystem();
                
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'tr-TR';
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                
                utterance.onstart = () => {
                    console.log('GPT konuşmaya başladı');
                    isSpeaking = true;
                    isGPTSpeaking = true;
                    updateStatus('GPT konuşuyor...', 'speaking');
                };
                
                utterance.onend = async () => {
                    console.log('GPT konuşması bitti');
                    isSpeaking = false;
                    isGPTSpeaking = false;
                    
                    // GPT konuşması bittikten sonra ses sistemini yeniden başlat
                    setTimeout(async () => {
                        if (!interviewEnded) {
                            await initAudio();
                            if (isAutoMode) {
                                startVoiceDetection();
                                updateStatus('Otomatik dinleme aktif', 'auto');
                            }
                        }
                    }, 2000); // 2 saniye bekle
                    
                    resolve();
                };
                
                utterance.onerror = async (event) => {
                    console.error('Konuşma hatası:', event.error);
                    isSpeaking = false;
                    isGPTSpeaking = false;
                    
                    setTimeout(async () => {
                        if (!interviewEnded) {
                            await initAudio();
                            if (isAutoMode) {
                                startVoiceDetection();
                                updateStatus('Otomatik dinleme aktif', 'auto');
                            }
                        }
                    }, 2000);
                    
                    resolve();
                };
                
                window.speechSynthesis.speak(utterance);
            });
        }

        async function initializeInterview() {
            try {
                // Mülakat kodunu kontrol et
                if (!interviewCode) {
                    showError('Geçersiz mülakat kodu');
                    return false;
                }

                // Ses sistemini başlat
                const initialized = await initAudio();
                if (!initialized) {
                    return false;
                }

                // UI'ı güncelle
                updateStatus('Mülakat başlatılıyor...', 'auto');
                document.getElementById('aivaAvatar').classList.remove('completed');

                return true;
            } catch (error) {
                console.error('Mülakat başlatma hatası:', error);
                showError('Mülakat başlatılamadı');
                return false;
            }
        }

        async function sendAudioToServer(audioBlob) {
            try {
                if (isSpeaking || isGPTSpeaking) {
                    console.log('GPT konuşuyor, ses kaydı işlenmeyecek');
                    return;
                }

                if (audioBlob.size === 0) {
                    console.log('Boş ses kaydı, işlem yapılmayacak');
                    return;
                }

                const formData = new FormData();
                formData.append('audio', audioBlob, 'audio.webm');
                formData.append('interview_code', interviewCode);

                console.log('Ses kaydı gönderiliyor...');
                const response = await fetch('/process_audio', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('Sunucu yanıt detayı:', errorText);
                    throw new Error(`HTTP error! status: ${response.status}, detail: ${errorText}`);
                }
                
                const data = await response.json();
                console.log('Sunucu yanıtı:', data);
                
                if (data.success) {
                    if (data.transcript) {
                        addMessageToChat('user', data.transcript);
                    }
                    if (data.response) {
                        addMessageToChat('assistant', data.response);
                        await speakResponse(data.response);
                    }
                    
                    if (data.interview_completed) {
                        interviewEnded = true;
                        await endInterview();
                    }
                } else {
                    showError(data.error || 'Ses işlenemedi');
                    
                    if (!data.continue_listening) {
                        isAutoMode = false;
                        updateStatus('Manuel moda geçildi (Space tuşunu kullanın)', 'manual');
                    }
                }
            } catch (error) {
                console.error('Sunucu hatası:', error);
                showError('Sunucu ile iletişim hatası: ' + error.message);
                
                // Hata durumunda ses sistemini yeniden başlat
                isMediaRecorderReady = false;
                if (isAutoMode) {
                    await initAudio();
                    startVoiceDetection();
                }
            }
        }

        // WebM'den WAV'a dönüştürme fonksiyonu
        async function convertToWav(webmBlob) {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await webmBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            const numberOfChannels = 1;
            const length = audioBuffer.length;
            const sampleRate = 16000;
            const buffer = audioContext.createBuffer(numberOfChannels, length, sampleRate);
            
            // Ses verilerini kopyala
            const channelData = audioBuffer.getChannelData(0);
            buffer.copyToChannel(channelData, 0);
            
            // WAV formatına dönüştür
            const wavData = audioBufferToWav(buffer);
            return new Blob([wavData], { type: 'audio/wav' });
        }

        // AudioBuffer'ı WAV formatına dönüştürme
        function audioBufferToWav(buffer) {
            const numberOfChannels = 1;
            const sampleRate = 16000;
            const format = 1; // PCM
            const bitDepth = 16;
            
            const bytesPerSample = bitDepth / 8;
            const blockAlign = numberOfChannels * bytesPerSample;
            
            const buffer32 = new Int32Array(44 + buffer.length * bytesPerSample);
            const view = new DataView(buffer32.buffer);
            
            // WAV header
            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + buffer.length * bytesPerSample, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, format, true);
            view.setUint16(22, numberOfChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * blockAlign, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitDepth, true);
            writeString(view, 36, 'data');
            view.setUint32(40, buffer.length * bytesPerSample, true);
            
            // Ses verilerini yaz
            const data = buffer.getChannelData(0);
            let offset = 44;
            for (let i = 0; i < data.length; i++, offset += 2) {
                const sample = Math.max(-1, Math.min(1, data[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            }
            
            return buffer32.buffer;
        }

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        function addMessageToChat(role, message) {
            const chatContainer = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-bubble ${role}-bubble`;
            messageDiv.textContent = message;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            if (role === 'assistant') {
                const avatar = document.getElementById('aivaAvatar');
                avatar.classList.add('speaking');
                setTimeout(() => avatar.classList.remove('speaking'), 1000);
            }
        }

        function showError(message) {
            // Hata mesajını göster
            const errorDiv = document.createElement('div');
            errorDiv.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
            errorDiv.textContent = message;
            document.body.appendChild(errorDiv);
            
            setTimeout(() => {
                errorDiv.remove();
            }, 3000);
        }

        async function endInterview() {
            try {
                await stopAudioSystem();
                
                console.log('Mülakat sonlandırma isteği gönderiliyor...');
                const response = await fetch('/stop_recording', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({
                        interview_code: interviewCode
                    })
                });
                
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('Sonlandırma yanıt detayı:', errorText);
                    throw new Error(`HTTP error! status: ${response.status}, detail: ${errorText}`);
                }
                
                const data = await response.json();
                console.log('Mülakat sonlandırma yanıtı:', data);
                
                if (data.success) {
                    addMessageToChat('assistant', 'Mülakat tamamlandı. Rapor oluşturuluyor...');
                    
                    // Raporu indir
                    if (data.report_url) {
                        const downloadLink = document.createElement('a');
                        downloadLink.href = data.report_url;
                        downloadLink.download = `mulakat_raporu_${new Date().toISOString().slice(0,10)}.pdf`;
                        downloadLink.click();
                    }
                    
                    // Webhook'a gönder
                    if (data.report_data) {
                        await sendReportToWebhook(data.report_data);
                    }
                    
                    showSuccess('Mülakat başarıyla tamamlandı ve rapor oluşturuldu!');
                    updateStatus('Mülakat tamamlandı', 'completed');
                    document.getElementById('aivaAvatar').classList.add('completed');
                    
                    // Rapor indirme butonu ekle
                    if (data.report_url) {
                        addDownloadButton(data.report_url);
                    }
                } else {
                    throw new Error(data.error || 'Rapor oluşturulamadı');
                }
                
            } catch (error) {
                console.error('Mülakat bitirme hatası:', error);
                showError('Mülakat sonlandırılırken bir hata oluştu: ' + error.message);
            }
        }

        async function sendReportToWebhook(reportData) {
            try {
                // Yeni webhook URL'si
                const webhookUrl = 'https://otomasyon.aivatech.io/api/v1/webhooks/B7iYtwVltWEzX2nvAaWCX';
                
                console.log('Webhook\'a gönderilen veri:', reportData);
                
                const response = await fetch(webhookUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({
                        interview_data: reportData,
                        timestamp: new Date().toISOString(),
                        status: 'completed'
                    })
                });
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Webhook hatası: ${response.status}, detay: ${errorText}`);
                }
                
                console.log('Rapor webhook\'a başarıyla gönderildi');
            } catch (error) {
                console.error('Webhook hatası:', error);
                showError('Rapor webhook\'a gönderilemedi: ' + error.message);
            }
        }

        function addDownloadButton(reportUrl) {
            const container = document.querySelector('.chat-container');
            const downloadDiv = document.createElement('div');
            downloadDiv.className = 'fixed bottom-4 right-4 flex gap-4';
            
            const downloadButton = document.createElement('button');
            downloadButton.className = 'bg-indigo-600 text-white px-6 py-3 rounded-lg shadow-lg hover:bg-indigo-700 transition-colors';
            downloadButton.innerHTML = '<i class="fas fa-download mr-2"></i>Raporu İndir';
            downloadButton.onclick = () => {
                const link = document.createElement('a');
                link.href = reportUrl;
                link.download = `mulakat_raporu_${new Date().toISOString().slice(0,10)}.pdf`;
                link.click();
            };
            
            downloadDiv.appendChild(downloadButton);
            container.appendChild(downloadDiv);
        }

        function showSuccess(message) {
            const successDiv = document.createElement('div');
            successDiv.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
            successDiv.textContent = message;
            document.body.appendChild(successDiv);
            
            setTimeout(() => {
                successDiv.remove();
            }, 5000);
        }

        // Modal işlemleri için yeni fonksiyonlar
        function showModal() {
            document.getElementById('infoModal').style.display = 'flex';
        }

        function closeModal() {
            document.getElementById('infoModal').style.display = 'none';
        }

        // Sayfa yüklendiğinde
        document.addEventListener('DOMContentLoaded', async () => {
            // Mülakatı başlat
            const initialized = await initializeInterview();
            if (!initialized) {
                return;
            }

            // Avatar tıklama - mod değiştirme
            document.getElementById('aivaAvatar').addEventListener('click', () => {
                if (!interviewEnded) {
                    isAutoMode = !isAutoMode;
                    if (isAutoMode) {
                        startVoiceDetection();
                        updateStatus('Otomatik dinleme aktif', 'auto');
                    } else {
                        updateStatus('Manuel mod aktif (Space tuşunu kullanın)', 'manual');
                    }
                }
            });
            
            // Space tuşu kontrolü
            document.addEventListener('keydown', (e) => {
                if (e.code === 'Space' && !isAutoMode) {
                    e.preventDefault();
                    if (!isRecording) startRecording();
                }
            });
            
            document.addEventListener('keyup', (e) => {
                if (e.code === 'Space' && !isAutoMode) {
                    e.preventDefault();
                    if (isRecording) stopRecording();
                }
            });

            // Sayfa yüklendiğinde modalı göster
            showModal();
        });
    </script>
</body>
</html> 