// ClipperVX Web App - Polling version
(function () {
    'use strict';

    // State
    const state = {
        videoInfo: null,
        jobId: null,
        localPath: null,
        isLocalFile: false,
        pollingInterval: null,
        antigravityAuth: false,
        settings: {
            geminiKey: localStorage.getItem('gemini_key') || '',
            openaiKey: localStorage.getItem('openai_key') || '',
            llmProvider: localStorage.getItem('llm_provider') || 'antigravity',
            llmModel: localStorage.getItem('llm_model') || 'claude-sonnet-4-5-thinking'
        }
    };

    // Elements
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const elements = {
        tabs: $$('.tab'),
        tabContents: $$('.tab-content'),
        urlInput: $('#url-input'),
        fetchBtn: $('#fetch-btn'),
        uploadZone: $('#upload-zone'),
        fileInput: $('#file-input'),
        uploadProgress: $('#upload-progress'),
        uploadFill: $('#upload-fill'),
        uploadStatus: $('#upload-status'),
        videoInfo: $('#video-info'),
        videoThumbnail: $('#video-thumbnail'),
        videoTitle: $('#video-title'),
        videoDuration: $('#video-duration'),
        videoIdDisplay: $('#video-id-display'),
        videoSource: $('#video-source'),
        optionsSection: $('#options-section'),
        qualityOption: $('#quality-option'),
        qualitySelect: $('#quality-select'),
        clipsCount: $('#clips-count'),
        minLength: $('#min-length'),
        maxLength: $('#max-length'),
        processBtn: $('#process-btn'),
        progressSection: $('#progress-section'),
        progressStage: $('#progress-stage'),
        progressFill: $('#progress-fill'),
        progressPercent: $('#progress-percent'),
        progressLog: $('#progress-log'),
        resultsSection: $('#results-section'),
        clipsGrid: $('#clips-grid'),
        themeToggle: $('#theme-toggle'),
        settingsBtn: $('#settings-btn'),
        settingsModal: $('#settings-modal'),
        settingsClose: $('#settings-close'),
        settingsCancel: $('#settings-cancel'),
        settingsSave: $('#settings-save'),
        geminiKey: $('#gemini-key'),
        openaiKey: $('#openai-key'),
        llmProvider: $('#llm-provider'),
        llmModel: $('#llm-model'),
        modelGroup: $('#model-group'),
        geminiKeyGroup: $('#gemini-key-group'),
        openaiKeyGroup: $('#openai-key-group'),
        antigravityAuthGroup: $('#antigravity-auth-group'),
        antigravityAuthBtn: $('#antigravity-auth-btn'),
        authStatus: $('#auth-status'),
        providerHint: $('#provider-hint'),
        connectionStatus: $('#connection-status')
    };

    // Initialize
    function init() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);

        elements.geminiKey.value = state.settings.geminiKey;
        elements.openaiKey.value = state.settings.openaiKey;
        elements.llmProvider.value = state.settings.llmProvider;

        // Update connection status
        updateConnectionStatus('connected');

        // Check Antigravity auth status
        checkAntigravityAuth();

        // Load models for selected provider
        loadModels(state.settings.llmProvider);
        updateProviderUI(state.settings.llmProvider);

        // Event listeners
        elements.tabs.forEach(tab => {
            tab.addEventListener('click', () => switchTab(tab.dataset.tab));
        });

        elements.fetchBtn.addEventListener('click', fetchVideoInfo);
        elements.urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') fetchVideoInfo();
        });

        elements.uploadZone.addEventListener('click', () => elements.fileInput.click());
        elements.fileInput.addEventListener('change', handleFileSelect);
        elements.uploadZone.addEventListener('dragover', handleDragOver);
        elements.uploadZone.addEventListener('dragleave', handleDragLeave);
        elements.uploadZone.addEventListener('drop', handleDrop);

        elements.processBtn.addEventListener('click', startProcessing);
        elements.themeToggle.addEventListener('click', toggleTheme);
        elements.settingsBtn.addEventListener('click', openSettings);
        elements.settingsClose.addEventListener('click', closeSettings);
        elements.settingsCancel.addEventListener('click', closeSettings);
        elements.settingsSave.addEventListener('click', saveSettings);
        elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === elements.settingsModal) closeSettings();
        });

        // Provider change handler
        elements.llmProvider.addEventListener('change', (e) => {
            const provider = e.target.value;
            loadModels(provider);
            updateProviderUI(provider);
        });

        // Antigravity auth button
        if (elements.antigravityAuthBtn) {
            elements.antigravityAuthBtn.addEventListener('click', authenticateAntigravity);
        }
    }

    async function checkAntigravityAuth() {
        try {
            const response = await fetch('/api/auth/antigravity/status');
            const data = await response.json();
            state.antigravityAuth = data.authenticated;
            updateAuthUI();
        } catch (e) {
            console.error('Failed to check Antigravity auth:', e);
        }
    }

    function updateAuthUI() {
        if (elements.authStatus) {
            elements.authStatus.textContent = state.antigravityAuth ?
                'Authenticated' : 'Not authenticated';
            elements.authStatus.style.color = state.antigravityAuth ? '#00ff88' : '#ff6b6b';
        }
        if (elements.antigravityAuthBtn) {
            const btnText = elements.antigravityAuthBtn.querySelector('#auth-btn-text');
            if (btnText) {
                btnText.textContent = state.antigravityAuth ?
                    'Re-authenticate' : 'Authenticate with Google';
            }
        }
    }

    async function authenticateAntigravity() {
        const btn = elements.antigravityAuthBtn;
        btn.disabled = true;
        btn.querySelector('#auth-btn-text').textContent = 'Authenticating...';

        try {
            const response = await fetch('/api/auth/antigravity', { method: 'POST' });
            const data = await response.json();

            if (data.status === 'authenticated' || data.status === 'already_authenticated') {
                state.antigravityAuth = true;
                updateAuthUI();
            } else {
                elements.authStatus.textContent = data.error || 'Authentication failed';
                elements.authStatus.style.color = '#ff6b6b';
            }
        } catch (e) {
            elements.authStatus.textContent = 'Error: ' + e.message;
            elements.authStatus.style.color = '#ff6b6b';
        } finally {
            btn.disabled = false;
            updateAuthUI();
        }
    }

    async function loadModels(provider) {
        if (provider === 'none') {
            if (elements.modelGroup) elements.modelGroup.style.display = 'none';
            return;
        }

        try {
            const response = await fetch(`/api/models?provider=${provider}`);
            const data = await response.json();

            if (elements.llmModel) {
                elements.llmModel.innerHTML = data.models.map(m =>
                    `<option value="${m.id}">${m.name}</option>`
                ).join('');

                // Restore saved model if available for this provider
                const savedModel = state.settings.llmModel;
                if (data.models.some(m => m.id === savedModel)) {
                    elements.llmModel.value = savedModel;
                }
            }
            if (elements.modelGroup) elements.modelGroup.style.display = 'block';
        } catch (e) {
            console.error('Failed to load models:', e);
        }
    }

    function updateProviderUI(provider) {
        // Show/hide relevant sections
        const hints = {
            'antigravity': 'Free access to Claude & Gemini via Google OAuth',
            'gemini': 'Requires Google Gemini API key',
            'openai': 'Requires OpenAI API key',
            'none': 'Uses heuristic-based clip selection'
        };

        if (elements.providerHint) {
            elements.providerHint.textContent = hints[provider] || '';
        }

        // Show/hide auth and key groups
        if (elements.antigravityAuthGroup) {
            elements.antigravityAuthGroup.style.display = provider === 'antigravity' ? 'block' : 'none';
        }
        if (elements.geminiKeyGroup) {
            elements.geminiKeyGroup.style.display = provider === 'gemini' ? 'block' : 'none';
        }
        if (elements.openaiKeyGroup) {
            elements.openaiKeyGroup.style.display = provider === 'openai' ? 'block' : 'none';
        }
        if (elements.modelGroup) {
            elements.modelGroup.style.display = provider !== 'none' ? 'block' : 'none';
        }
    }

    function switchTab(tabName) {
        elements.tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
        elements.tabContents.forEach(c => c.classList.toggle('active', c.id === tabName + '-tab'));
        state.videoInfo = null;
        state.localPath = null;
        state.isLocalFile = tabName === 'local';
        elements.videoInfo.classList.add('hidden');
        elements.optionsSection.classList.add('hidden');
    }

    function updateConnectionStatus(status) {
        const statusEl = elements.connectionStatus;
        statusEl.className = 'connection-status ' + status;
        const text = statusEl.querySelector('.status-text');
        switch (status) {
            case 'connecting': text.textContent = 'Connecting...'; break;
            case 'connected': text.textContent = 'Ready';
                setTimeout(() => statusEl.classList.add('fade'), 2000);
                break;
            case 'disconnected': text.textContent = 'Disconnected'; break;
            case 'error': text.textContent = 'Error'; break;
        }
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    }

    function openSettings() { elements.settingsModal.classList.remove('hidden'); }
    function closeSettings() { elements.settingsModal.classList.add('hidden'); }

    function saveSettings() {
        state.settings.geminiKey = elements.geminiKey.value;
        state.settings.openaiKey = elements.openaiKey.value;
        state.settings.llmProvider = elements.llmProvider.value;
        state.settings.llmModel = elements.llmModel ? elements.llmModel.value : '';
        localStorage.setItem('gemini_key', state.settings.geminiKey);
        localStorage.setItem('openai_key', state.settings.openaiKey);
        localStorage.setItem('llm_provider', state.settings.llmProvider);
        localStorage.setItem('llm_model', state.settings.llmModel);
        closeSettings();
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        elements.uploadZone.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        elements.uploadZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        elements.uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            uploadFile(files[0]);
        } else {
            alert('Please drop a video file');
        }
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) uploadFile(file);
    }

    async function uploadFile(file) {
        elements.uploadZone.classList.add('hidden');
        elements.uploadProgress.classList.remove('hidden');

        const formData = new FormData();
        formData.append('video', file);

        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = (e.loaded / e.total) * 100;
                elements.uploadFill.style.width = percent + '%';
                elements.uploadStatus.textContent = `Uploading... ${Math.round(percent)}%`;
            }
        });

        xhr.onload = function () {
            elements.uploadProgress.classList.add('hidden');
            elements.uploadZone.classList.remove('hidden');
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                state.videoInfo = data;
                state.localPath = data.path;
                state.isLocalFile = true;
                displayVideoInfo(data, true);
            } else {
                alert('Upload failed');
            }
        };

        xhr.onerror = function () {
            elements.uploadProgress.classList.add('hidden');
            elements.uploadZone.classList.remove('hidden');
            alert('Upload failed');
        };

        xhr.open('POST', '/api/upload');
        xhr.send(formData);
    }

    async function fetchVideoInfo() {
        const url = elements.urlInput.value.trim();
        if (!url) return;

        elements.fetchBtn.disabled = true;
        elements.fetchBtn.innerHTML = '<span>Loading...</span>';

        try {
            const response = await fetch('/api/video-info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });

            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }

            state.videoInfo = data;
            state.isLocalFile = false;
            state.localPath = null;
            displayVideoInfo(data, false);
        } catch (error) {
            alert('Failed to fetch video info: ' + error.message);
        } finally {
            elements.fetchBtn.disabled = false;
            elements.fetchBtn.innerHTML = '<span>Fetch</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
        }
    }

    function displayVideoInfo(data, isLocal) {
        if (data.thumbnail) {
            elements.videoThumbnail.src = data.thumbnail;
            elements.videoThumbnail.style.display = 'block';
        } else {
            elements.videoThumbnail.style.display = 'none';
        }

        elements.videoTitle.textContent = data.title;
        elements.videoDuration.textContent = formatDuration(data.duration);
        elements.videoIdDisplay.textContent = `ID: ${data.video_id}`;
        elements.videoSource.textContent = isLocal ? '📁 Local File' : '🎬 YouTube';

        if (isLocal || !data.formats || data.formats.length === 0) {
            elements.qualityOption.classList.add('hidden');
        } else {
            elements.qualityOption.classList.remove('hidden');
            elements.qualitySelect.innerHTML = '<option value="best">Best Available</option>';
            data.formats.forEach(fmt => {
                const option = document.createElement('option');
                option.value = fmt.format_id;
                option.textContent = `${fmt.resolution} (${fmt.ext})`;
                elements.qualitySelect.appendChild(option);
            });
        }

        elements.videoInfo.classList.remove('hidden');
        elements.optionsSection.classList.remove('hidden');
    }

    function formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    async function startProcessing() {
        if (!state.videoInfo) return;

        elements.processBtn.disabled = true;
        elements.processBtn.innerHTML = '<span>Processing...</span>';
        elements.progressSection.classList.remove('hidden');
        elements.resultsSection.classList.add('hidden');
        elements.progressLog.innerHTML = '';
        elements.progressFill.style.width = '0%';
        elements.progressPercent.textContent = '0%';

        try {
            const payload = {
                clips: parseInt(elements.clipsCount.value),
                min_length: parseInt(elements.minLength.value),
                max_length: parseInt(elements.maxLength.value),
                gemini_key: state.settings.geminiKey,
                openai_key: state.settings.openaiKey,
                llm_provider: state.settings.llmProvider,
                llm_model: state.settings.llmModel,
                video_id: state.videoInfo.video_id
            };

            if (state.isLocalFile) {
                payload.local_path = state.localPath;
            } else {
                payload.url = elements.urlInput.value.trim();
                payload.quality = elements.qualitySelect.value;
            }

            const response = await fetch('/api/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (data.error) {
                showError(data.error);
                return;
            }

            state.jobId = data.job_id;
            addLog('Job started: ' + data.job_id);

            // Start polling for status
            startPolling();
        } catch (error) {
            showError(error.message);
        }
    }

    function startPolling() {
        if (state.pollingInterval) clearInterval(state.pollingInterval);

        let lastStage = '';
        let lastPercent = 0;

        state.pollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/job/${state.jobId}`);
                const data = await response.json();

                if (data.status === 'processing') {
                    // progress already comes as 0-100 from server
                    const percent = data.progress;
                    updateProgress(data.stage, percent);

                    // Log stage changes
                    if (data.stage !== lastStage) {
                        addLog(`📌 ${data.stage}`);
                        lastStage = data.stage;
                    }
                } else if (data.status === 'complete') {
                    clearInterval(state.pollingInterval);
                    updateProgress('Complete!', 100);
                    showResults(data.clips, data.errors, data.video_id);
                } else if (data.status === 'error') {
                    clearInterval(state.pollingInterval);
                    showError(data.error);
                }
            } catch (e) {
                console.error('Polling error:', e);
            }
        }, 1000);
    }

    function updateProgress(stage, percent) {
        elements.progressStage.textContent = stage;
        elements.progressFill.style.width = `${percent}%`;
        elements.progressPercent.textContent = `${percent}%`;
    }

    function addLog(message) {
        const time = new Date().toLocaleTimeString();
        elements.progressLog.innerHTML += `<div>[${time}] ${message}</div>`;
        elements.progressLog.scrollTop = elements.progressLog.scrollHeight;
    }

    function showResults(clips, errors, videoId) {
        elements.processBtn.disabled = false;
        elements.processBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Generate Clips</span>';

        elements.resultsSection.classList.remove('hidden');
        elements.clipsGrid.innerHTML = '';

        clips.forEach(clip => {
            const relativePath = clip.relative_path || `${videoId}/${clip.filename}`;

            const card = document.createElement('div');
            card.className = 'clip-card';
            card.innerHTML = `
                <video class="clip-video" controls preload="metadata">
                    <source src="/output/${relativePath}" type="video/mp4">
                </video>
                <div class="clip-info">
                    <div class="clip-name" title="${clip.filename}">${clip.filename}</div>
                    <div class="clip-hook">${clip.hook || ''}</div>
                </div>
                <div class="clip-actions">
                    <a href="/output/${relativePath}" download="${clip.filename}" class="btn-secondary">Download</a>
                </div>
            `;
            elements.clipsGrid.appendChild(card);
        });

        if (errors && errors.length > 0) {
            errors.forEach(err => addLog('⚠️ ' + err));
        }

        addLog('✅ Processing complete!');
    }

    function showError(message) {
        elements.processBtn.disabled = false;
        elements.processBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Generate Clips</span>';
        addLog('❌ Error: ' + message);
        alert('Error: ' + message);
    }

    document.addEventListener('DOMContentLoaded', init);
})();
