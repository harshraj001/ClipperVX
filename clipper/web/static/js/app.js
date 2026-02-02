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
        },
        youtubeAuth: false,
        youtubeChannel: null
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
        fontSelect: $('#font-select'),
        fontPreview: $('#font-preview'),
        processBtn: $('#process-btn'),
        progressSection: $('#progress-section'),
        progressStage: $('#progress-stage'),
        progressFill: $('#progress-fill'),
        progressPercent: $('#progress-percent'),
        progressLog: $('#progress-log'),
        stopBtn: $('#stop-btn'),
        resultsSection: $('#results-section'),
        clipsGrid: $('#clips-grid'),
        historySection: $('#history-section'),
        historyList: $('#history-list'),
        clearHistoryBtn: $('#clear-history-btn'),
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
        connectionStatus: $('#connection-status'),
        detailsModal: $('#details-modal'),
        detailsClose: $('#details-close'),
        detailsCloseBtn: $('#details-close-btn'),
        clipTitle: $('#clip-title'),
        clipDescription: $('#clip-description'),
        clipHashtags: $('#clip-hashtags'),
        copyTitleBtn: $('#copy-title-btn'),
        copyDescriptionBtn: $('#copy-description-btn'),
        copyHashtagsBtn: $('#copy-hashtags-btn'),
        youtubeAuthBtn: $('#youtube-auth-btn'),
        youtubeAuthStatus: $('#youtube-auth-status'),
        postAllYoutubeBtn: $('#post-all-youtube-btn'),
        youtubePostBtn: $('#youtube-post-btn')
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

        // Load available fonts
        loadFonts();

        // Load job history
        loadJobHistory();

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

        // Font preview update
        if (elements.fontSelect) {
            elements.fontSelect.addEventListener('change', updateFontPreview);
        }

        // Stop button
        if (elements.stopBtn) {
            elements.stopBtn.addEventListener('click', stopCurrentJob);
        }

        // Clear history button
        if (elements.clearHistoryBtn) {
            elements.clearHistoryBtn.addEventListener('click', clearHistory);
        }

        // Clip Details Modal listeners
        if (elements.detailsClose) {
            elements.detailsClose.addEventListener('click', () => {
                elements.detailsModal.classList.add('hidden');
            });
        }
        if (elements.detailsCloseBtn) {
            elements.detailsCloseBtn.addEventListener('click', () => {
                elements.detailsModal.classList.add('hidden');
            });
        }
        if (elements.detailsModal) {
            elements.detailsModal.addEventListener('click', (e) => {
                if (e.target === elements.detailsModal) {
                    elements.detailsModal.classList.add('hidden');
                }
            });
        }

        // Copy buttons
        if (elements.copyTitleBtn) {
            elements.copyTitleBtn.addEventListener('click', () => copyToClipboard(elements.clipTitle, elements.copyTitleBtn));
        }
        if (elements.copyDescriptionBtn) {
            elements.copyDescriptionBtn.addEventListener('click', () => copyToClipboard(elements.clipDescription, elements.copyDescriptionBtn));
        }
        if (elements.copyHashtagsBtn) {
            elements.copyHashtagsBtn.addEventListener('click', () => copyToClipboard(elements.clipHashtags, elements.copyHashtagsBtn));
        }

        // YouTube Automation
        if (elements.youtubeAuthBtn) {
            elements.youtubeAuthBtn.addEventListener('click', connectYouTube);
        }
        if (elements.postAllYoutubeBtn) {
            elements.postAllYoutubeBtn.addEventListener('click', () => postAllClips());
        }

        // Initial YouTube check
        checkYouTubeStatus();
    }

    // Copy helper
    function copyToClipboard(inputElement, btnElement) {
        inputElement.select();
        inputElement.setSelectionRange(0, 99999); // Mobile compatibility
        navigator.clipboard.writeText(inputElement.value).then(() => {
            const originalIcon = btnElement.innerHTML;
            btnElement.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
            setTimeout(() => {
                btnElement.innerHTML = originalIcon;
            }, 1500);
        });
    }

    // Show clip details modal
    // Show clip details modal
    function showClipDetails(clip) {
        elements.clipTitle.value = clip.title || 'Generating title... (or generation failed)';
        elements.clipDescription.value = clip.description || 'Generating description...';

        let hashtags = '';
        if (Array.isArray(clip.hashtags) && clip.hashtags.length > 0) {
            hashtags = clip.hashtags.join(' ');
        } else if (typeof clip.hashtags === 'string' && clip.hashtags.length > 0) {
            hashtags = clip.hashtags;
        } else {
            hashtags = '#viral #shorts';
        }
        elements.clipHashtags.value = hashtags;

        elements.detailsModal.classList.remove('hidden');
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

    // YouTube Automation Logic
    async function checkYouTubeStatus() {
        try {
            const response = await fetch('/api/auth/youtube/status');
            const data = await response.json();

            state.youtubeAuth = data.authenticated;
            if (data.channel) {
                state.youtubeChannel = data.channel;
            }
            updateYouTubeUI();
        } catch (e) {
            console.error('Failed to check YouTube status:', e);
        }
    }

    function updateYouTubeUI() {
        if (elements.youtubeAuthBtn && elements.youtubeAuthStatus) {
            if (state.youtubeAuth) {
                const channelName = state.youtubeChannel ? state.youtubeChannel.title : 'Connected';
                elements.youtubeAuthBtn.textContent = 'Connected as ' + channelName;
                elements.youtubeAuthBtn.disabled = true;
                elements.youtubeAuthBtn.classList.add('btn-success');
                elements.youtubeAuthStatus.textContent = 'Ready to upload clips';
                elements.youtubeAuthStatus.style.color = '#00ff88';
            } else {
                elements.youtubeAuthBtn.textContent = 'Connect YouTube Account';
                elements.youtubeAuthBtn.disabled = false;
                elements.youtubeAuthStatus.textContent = 'Link your channel to post clips directly';
                elements.youtubeAuthStatus.style.color = '';
            }
        }
    }

    async function connectYouTube() {
        const btn = elements.youtubeAuthBtn;
        btn.disabled = true;
        btn.textContent = 'Connecting...';

        try {
            const response = await fetch('/api/auth/youtube/url');
            const data = await response.json();

            if (data.url) {
                window.location.href = data.url;
            } else {
                alert('Failed to get auth URL');
                btn.disabled = false;
                btn.textContent = 'Connect YouTube Account';
            }
        } catch (e) {
            console.error('Auth error:', e);
            btn.disabled = false;
            btn.textContent = 'Connect YouTube Account';
        }
    }

    async function postClip(clipIndex, btn) {
        if (!confirm('Are you sure you want to post this clip to YouTube?')) return;

        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.textContent = 'Posting...';

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: state.jobId,
                    clip_index: clipIndex
                })
            });
            const data = await response.json();

            if (data.status === 'success') {
                btn.textContent = 'Posted!';
                btn.classList.add('btn-success');
                addLog(`✅ Clip posted to YouTube (ID: ${data.video_id})`);
            } else {
                alert('Error: ' + data.error);
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        } catch (e) {
            alert('Upload failed: ' + e.message);
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }

    async function postAllClips() {
        if (!state.jobId) return;
        if (!confirm('This will schedule all clips to be posted starting 1 hour from now, 1 hour apart. Proceed?')) return;

        const btn = elements.postAllYoutubeBtn;
        btn.disabled = true;
        btn.textContent = 'Scheduling...';

        try {
            const response = await fetch('/api/upload_all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: state.jobId })
            });
            const data = await response.json();

            if (data.status === 'success') {
                btn.textContent = 'Scheduled!';
                btn.classList.add('btn-success');
                addLog(`✅ ${data.uploads.length} clips scheduled for YouTube upload`);

                // Show schedule details in log
                data.uploads.forEach(u => {
                    const time = new Date(u.scheduled_for).toLocaleString();
                    addLog(`📅 Scheduled: ${u.filename} at ${time}`);
                });
            } else {
                alert('Error: ' + data.error);
                btn.disabled = false;
                btn.textContent = 'Post All (Scheduled)';
            }
        } catch (e) {
            alert('Bulk upload failed: ' + e.message);
            btn.disabled = false;
            btn.textContent = 'Post All (Scheduled)';
        }
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
        if (elements.stopBtn) elements.stopBtn.style.display = '';

        try {
            const payload = {
                clips: parseInt(elements.clipsCount.value),
                min_length: parseInt(elements.minLength.value),
                max_length: parseInt(elements.maxLength.value),
                gemini_key: state.settings.geminiKey,
                openai_key: state.settings.openaiKey,
                llm_provider: state.settings.llmProvider,
                llm_model: state.settings.llmModel,
                video_id: state.videoInfo.video_id,
                title: state.videoInfo.title || 'Video',
                font: elements.fontSelect ? elements.fontSelect.value : 'Komika Axis'
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
                } else if (data.status === 'stopped') {
                    clearInterval(state.pollingInterval);
                    console.log('Job stopped');
                }
            } catch (e) {
                console.error('Polling error:', e);
            }
        }, 1000);
    }

    function stopPolling() {
        if (state.pollingInterval) {
            clearInterval(state.pollingInterval);
            state.pollingInterval = null;
        }
        elements.processBtn.disabled = false;
        elements.processBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Generate Clips</span>';
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

        // Hide progress and show results
        elements.progressSection.classList.add('hidden');
        elements.resultsSection.classList.remove('hidden');
        elements.clipsGrid.innerHTML = '';

        // Show/Hide Post All button
        if (elements.postAllYoutubeBtn) {
            elements.postAllYoutubeBtn.style.display = state.youtubeAuth ? 'flex' : 'none';
        }

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
                <div class="clip-actions" style="display: flex; gap: 8px;">
                    <button class="btn-primary view-details-btn" style="flex: 1;">Clip Details</button>
                    <a href="/output/${relativePath}" download="${clip.filename}" class="btn-secondary" style="flex: 1; text-align: center;">Download</a>
                </div>
                <button class="btn-secondary youtube-single-btn" style="margin-top: 8px; width: 100%; display: flex; align-items: center; justify-content: center; gap: 6px;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M23.5 6.19a3.02 3.02 0 0 0-2.12-2.14C19.54 3.5 12 3.5 12 3.5s-7.54 0-9.38.55A3.02 3.02 0 0 0 .5 6.19 31.7 31.7 0 0 0 0 12a31.7 31.7 0 0 0 .5 5.81 3.02 3.02 0 0 0 2.12 2.14c1.84.55 9.38.55 9.38.55s7.54 0 9.38-.55a3.02 3.02 0 0 0 2.12-2.14A31.7 31.7 0 0 0 24 12a31.7 31.7 0 0 0-.5-5.81zM9.54 15.57V8.43L15.82 12l-6.28 3.57z" />
                    </svg>
                    Post to YouTube
                </button>
            `;

            // Add listener
            const detailsBtn = card.querySelector('.view-details-btn');
            detailsBtn.addEventListener('click', () => {
                showClipDetails(clip);

                // Update the "Post to YouTube" button inside the modal
                if (elements.youtubePostBtn) {
                    // Remove old listeners by cloning
                    const newBtn = elements.youtubePostBtn.cloneNode(true);
                    elements.youtubePostBtn.parentNode.replaceChild(newBtn, elements.youtubePostBtn);
                    elements.youtubePostBtn = newBtn;

                    elements.youtubePostBtn.addEventListener('click', () => {
                        postClip(clips.indexOf(clip), elements.youtubePostBtn);
                    });

                    // Update state
                    if (!state.youtubeAuth) {
                        elements.youtubePostBtn.style.display = 'none';
                    } else {
                        elements.youtubePostBtn.style.display = 'flex';
                    }
                }
            });

            // Single post button listener
            const postBtn = card.querySelector('.youtube-single-btn');
            if (!state.youtubeAuth) {
                postBtn.style.display = 'none';
            } else {
                postBtn.addEventListener('click', () => postClip(clips.indexOf(clip), postBtn));
            }

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

    // Load available fonts
    async function loadFonts() {
        try {
            const response = await fetch('/api/fonts');
            const data = await response.json();

            if (elements.fontSelect && data.fonts) {
                elements.fontSelect.innerHTML = '';

                // Create style element for fonts
                let fontCss = '';

                data.fonts.forEach(font => {
                    if (font.available) {
                        const option = document.createElement('option');
                        option.value = font.name;
                        option.textContent = font.name;
                        elements.fontSelect.appendChild(option);

                        // Add font-face rule
                        if (font.path) {
                            fontCss += `
                                @font-face {
                                    font-family: '${font.name}';
                                    src: url('${font.path}');
                                }
                            `;
                        }
                    }
                });

                // Inject font styles
                if (fontCss) {
                    const style = document.createElement('style');
                    style.textContent = fontCss;
                    document.head.appendChild(style);
                }

                // Set saved font
                const savedFont = localStorage.getItem('caption_font');
                if (savedFont) {
                    elements.fontSelect.value = savedFont;
                }

                updateFontPreview();
            }
        } catch (e) {
            console.error('Failed to load fonts:', e);
        }
    }

    // Update font preview
    function updateFontPreview() {
        if (elements.fontSelect && elements.fontPreview) {
            const selectedFont = elements.fontSelect.value;
            localStorage.setItem('caption_font', selectedFont);
            elements.fontPreview.innerHTML = `<span style="font-family: '${selectedFont}', sans-serif; font-size: 24px;">SAMPLE TEXT Preview</span>`;
        }
    }

    // Stop current job
    async function stopCurrentJob() {
        if (!state.jobId) return;

        try {
            const response = await fetch(`/api/job/${state.jobId}/stop`, { method: 'POST' });
            const data = await response.json();

            if (data.status === 'stopped') {
                addLog('🛑 Job stopped by user');
                stopPolling();
                elements.progressStage.textContent = 'Stopped';
                if (elements.stopBtn) {
                    elements.stopBtn.style.display = 'none';
                }
            }
        } catch (e) {
            console.error('Failed to stop job:', e);
        }
    }

    async function clearHistory() {
        if (!confirm('Are you sure you want to clear your job history? This cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch('/api/jobs', { method: 'DELETE' });
            const data = await response.json();

            if (data.status === 'cleared') {
                elements.historySection.classList.add('hidden');
                elements.historyList.innerHTML = '';
            }
        } catch (e) {
            console.error('Failed to clear history:', e);
            alert('Failed to clear history');
        }
    }

    // Load job history
    async function loadJobHistory() {
        try {
            const response = await fetch('/api/jobs');
            const data = await response.json();

            if (data.jobs && data.jobs.length > 0) {
                elements.historySection.classList.remove('hidden');
                elements.historyList.innerHTML = '';

                // Sort by created time descending
                const sortedJobs = data.jobs.sort((a, b) => (b.created || 0) - (a.created || 0));

                // Show recent 5 jobs
                sortedJobs.slice(0, 5).forEach(job => {
                    const item = document.createElement('div');
                    item.className = 'history-item';

                    const created = job.created ? new Date(job.created * 1000).toLocaleString() : 'Unknown';

                    let statusText = job.status;
                    if (job.status === 'processing' && job.progress !== undefined) {
                        statusText = `Processing (${Math.round(job.progress)}%)`;
                    }

                    item.innerHTML = `
                        <div class="history-item-info">
                            <div class="history-item-title">${job.title || job.video_id || job.id}</div>
                            <div class="history-item-meta">${created} • Font: ${job.font || 'Default'}</div>
                        </div>
                        <span class="history-item-status ${job.status}">${statusText}</span>
                    `;

                    // Click to view results if complete
                    if (job.status === 'complete' && job.clips) {
                        item.style.cursor = 'pointer';
                        item.addEventListener('click', () => {
                            showResults(job.clips, job.errors || [], job.video_id);
                            elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
                        });
                    }

                    // Click to resume monitoring if processing
                    if (job.status === 'processing' || job.status === 'started') {
                        item.style.cursor = 'pointer';
                        item.title = "Click to view progress";
                        item.addEventListener('click', () => {
                            state.jobId = job.id;
                            state.videoInfo = {
                                video_id: job.video_id,
                                title: job.title
                            };

                            // Restore UI state
                            elements.progressSection.classList.remove('hidden');
                            elements.resultsSection.classList.add('hidden');
                            elements.optionsSection.classList.add('hidden');
                            elements.optionsSection.classList.add('hidden');
                            elements.processBtn.disabled = true;
                            elements.processBtn.innerHTML = '<span class="spinner"></span> Processing...';
                            if (elements.stopBtn) elements.stopBtn.style.display = '';

                            elements.progressSection.scrollIntoView({ behavior: 'smooth' });
                            startPolling();
                        });
                    }

                    elements.historyList.appendChild(item);
                });
            } else {
                elements.historySection.classList.add('hidden');
            }
        } catch (e) {
            console.error('Failed to load job history:', e);
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
