// HTML Element references
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const resultSection = document.getElementById('result-section');
const progressBar = document.getElementById('progress-bar');
const steps = [
    document.getElementById('step-1'),
    document.getElementById('step-2'),
    document.getElementById('step-3'),
    document.getElementById('step-4')
];

let backendResultData = null;

// Three.js Global Variables
let scene, camera, renderer, controls;
let airwayMesh = null;
let highlightLine = null;
let contourSlicesData = [];
let smoothedContourSlicesData = [];
let sliceAreasData = [];
let referenceMesh = null;
let arrowHelpers = [];
let isPlaying3D = true;
let currentPlaybackFrame = 0;
let playbackSpeedMs = 80;
let playbackInterval = null;

// Initialize 3D Scene on load
init3DScene();

// Handle File Drop and Selection
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        startProcessing();
    }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) startProcessing();
});

// Slider and Toggles Logic
const slider = document.getElementById('xai-slider');
const heatmapImg = document.getElementById('heatmap-img');
const contourImg = document.getElementById('contour-img');
const collapseImg = document.getElementById('collapse-img');
const sliderLine = document.querySelector('.slider-line');
const toggleHeatmap = document.getElementById('toggle-heatmap');
const toggleContour = document.getElementById('toggle-contour');

slider.addEventListener('input', (e) => {
    const value = e.target.value; 
    if (collapseImg) {
        collapseImg.style.clipPath = `polygon(${value}% 0, 100% 0, 100% 100%, ${value}% 100%)`;
    }
    heatmapImg.style.clipPath = `polygon(${value}% 0, 100% 0, 100% 100%, ${value}% 100%)`;
    contourImg.style.clipPath = `polygon(${value}% 0, 100% 0, 100% 100%, ${value}% 100%)`;
    sliderLine.style.left = `${value}%`;
});

toggleHeatmap.addEventListener('change', (e) => {
    heatmapImg.style.opacity = e.target.checked ? '1' : '0';
});
toggleContour.addEventListener('change', (e) => {
    contourImg.style.opacity = e.target.checked ? '1' : '0';
});

// Playback Timeline Controls Event Listeners
document.getElementById('btn-play-pause').addEventListener('click', () => {
    isPlaying3D = !isPlaying3D;
    const btn = document.getElementById('btn-play-pause');
    btn.textContent = isPlaying3D ? 'Pause' : 'Play';
    btn.style.background = isPlaying3D ? 'var(--primary)' : '#10b981';
});

document.getElementById('timeline-slider').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    // Pause automatic playback on manual slide drag
    if (isPlaying3D) {
        isPlaying3D = false;
        const btn = document.getElementById('btn-play-pause');
        btn.textContent = 'Play';
        btn.style.background = '#10b981';
    }
    updateUIForFrame(val);
});

document.getElementById('select-speed').addEventListener('change', (e) => {
    playbackSpeedMs = parseInt(e.target.value);
    startPlaybackLoop();
});

// Initialize Three.js scene, camera, lights, and renderer
function init3DScene() {
    const container = document.getElementById('canvas-3d');
    if (!container) return;

    // 1. Create Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf1f5f9); // Clean medical light gray

    // 2. Create Camera
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 18); // Default perspective matching 2D view

    // 3. Create WebGL Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // 4. Setup Controls (Rotate, Zoom, Pan)
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 80;
    controls.target.set(0, 0, 0);

    // 5. Add Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(10, 25, 15);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight2.position.set(-10, -15, -10);
    scene.add(dirLight2);

    // Handle container resize
    window.addEventListener('resize', onWindowResize);

    // Start rendering frame loop
    animate();
}

function onWindowResize() {
    const container = document.getElementById('canvas-3d');
    if (!container || !camera || !renderer) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

// Generate color based on lumen area fraction
function getColorForRatio(ratio) {
    // ratio: 0.0 (fully collapsed, red) to 1.0 (fully open, green)
    let r, g, b;
    if (ratio < 0.5) {
        // Red to Yellow
        let t = ratio / 0.5;
        r = 1.0;
        g = t;
        b = 0.0;
    } else {
        // Yellow to Green
        let t = (ratio - 0.5) / 0.5;
        r = 1.0 - t;
        g = 1.0;
        b = 0.0;
    }
    return new THREE.Color(r, g, b);
}

// Reconstruct 3D Airway Tube from slices
function render3DAirway(contourSlices, sliceAreas) {
    // Clear old elements from the scene
    if (airwayMesh) {
        scene.remove(airwayMesh);
        airwayMesh.geometry.dispose();
        airwayMesh.material.dispose();
        airwayMesh = null;
    }
    if (referenceMesh) {
        scene.remove(referenceMesh);
        referenceMesh.geometry.dispose();
        referenceMesh.material.dispose();
        referenceMesh = null;
    }
    if (highlightLine) {
        scene.remove(highlightLine);
        highlightLine.geometry.dispose();
        highlightLine = null;
    }

    const N = contourSlices.length; // Number of slices in data (40)
    const M = 32; // Number of points per slice
    const scale = 0.07; // Scaling factor for coordinates
    const ySpacing_render = 1.0; // Spatial Y height spacing
    const slices_count = 5; // We render a short hollow ring with 5 layers

    // temporal smoothing of contours (moving average window of 5 slices)
    const smoothedSlices = [];
    const windowSize = 5;
    const halfWin = Math.floor(windowSize / 2);
    
    for (let i = 0; i < N; i++) {
        const smoothedSlice = [];
        for (let j = 0; j < M; j++) {
            let sumX = 0;
            let sumY = 0;
            let count = 0;
            for (let w = -halfWin; w <= halfWin; w++) {
                const idx = i + w;
                if (idx >= 0 && idx < N) {
                    sumX += contourSlices[idx][j][0];
                    sumY += contourSlices[idx][j][1];
                    count++;
                }
            }
            smoothedSlice.push([sumX / count, sumY / count]);
        }
        smoothedSlices.push(smoothedSlice);
    }
    
    // Save to global smoothed slices data for highlight synchronization
    smoothedContourSlicesData = smoothedSlices;

    const maxIdx = sliceAreas.indexOf(Math.max(...sliceAreas));
    const maxSlice = smoothedSlices[maxIdx];

    // Calculate centroid of maxSlice to center the referenceMesh around the origin
    let maxCx = 0, maxCz = 0;
    for (let j = 0; j < M; j++) {
        maxCx += maxSlice[j][0];
        maxCz += maxSlice[j][1];
    }
    maxCx /= M;
    maxCz /= M;

    // Build the reference envelope using the maximum coordinate distance from centroid
    // across all slices to ensure the dynamic mesh is always fully enclosed.
    const referenceSlice = [];
    for (let j = 0; j < M; j++) {
        let maxDist = 0;
        let bestPt = [maxSlice[j][0], maxSlice[j][1]];
        
        for (let i = 0; i < N; i++) {
            const slice_i = smoothedSlices[i];
            
            // Centroid of slice i
            let cx_i = 0, cz_i = 0;
            for (let k = 0; k < M; k++) {
                cx_i += slice_i[k][0];
                cz_i += slice_i[k][1];
            }
            cx_i /= M;
            cz_i /= M;
            
            const dx = slice_i[j][0] - cx_i;
            const dz = slice_i[j][1] - cz_i;
            const dist = Math.sqrt(dx*dx + dz*dz);
            if (dist > maxDist) {
                maxDist = dist;
                bestPt = [maxCx + dx, maxCz + dz]; // project onto max centroid
            }
        }
        referenceSlice.push(bestPt);
    }

    // Build Translucent Outer Reference Envelope (Patient's Max-Open airway silhouette)
    const refVertices = [];
    const refIndices = [];
    for (let i = 0; i < slices_count; i++) {
        const zHeight = (i - Math.floor(slices_count / 2)) * ySpacing_render;
        for (let j = 0; j < M; j++) {
            const pt = referenceSlice[j];
            const x = (pt[0] - maxCx) * scale;
            const y = -(pt[1] - maxCz) * scale;
            refVertices.push(x, y, zHeight);
        }
    }

    for (let i = 0; i < slices_count - 1; i++) {
        for (let j = 0; j < M; j++) {
            const p1 = i * M + j;
            const p2 = i * M + ((j + 1) % M);
            const p3 = (i + 1) * M + j;
            const p4 = (i + 1) * M + ((j + 1) % M);
            refIndices.push(p1, p3, p2);
            refIndices.push(p2, p3, p4);
        }
    }

    const refGeometry = new THREE.BufferGeometry();
    refGeometry.setAttribute('position', new THREE.Float32BufferAttribute(refVertices, 3));
    refGeometry.setIndex(refIndices);
    refGeometry.computeVertexNormals();

    const refMaterial = new THREE.MeshPhongMaterial({
        color: 0x334155, // Faint grey-blue reference outline
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.12, // Highly transparent outer envelope
        flatShading: false
    });

    referenceMesh = new THREE.Mesh(refGeometry, refMaterial);
    scene.add(referenceMesh);

    // Build the dynamic airway mesh (initially filled with empty coordinates)
    const vertices = [];
    const colors = [];
    const indices = [];

    for (let i = 0; i < slices_count; i++) {
        const zHeight = (i - Math.floor(slices_count / 2)) * ySpacing_render;
        for (let j = 0; j < M; j++) {
            vertices.push(0, 0, zHeight); // Placeholder positions, will be updated in updateHighlightSlice
            colors.push(0, 1, 0); // Placeholder colors (green)
        }
    }

    for (let i = 0; i < slices_count - 1; i++) {
        for (let j = 0; j < M; j++) {
            const p1 = i * M + j;
            const p2 = i * M + ((j + 1) % M);
            const p3 = (i + 1) * M + j;
            const p4 = (i + 1) * M + ((j + 1) % M);
            indices.push(p1, p3, p2);
            indices.push(p2, p3, p4);
        }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        shininess: 90,
        specular: 0x333333,
        transparent: true,
        opacity: 0.85,
        flatShading: false
    });

    airwayMesh = new THREE.Mesh(geometry, material);
    scene.add(airwayMesh);
}

// Highlight the contour shape of a specific slice (frame) in 3D
// Highlight the contour shape of a specific slice (frame) in 3D
function updateHighlightSlice(idx) {
    if (!smoothedContourSlicesData || smoothedContourSlicesData.length === 0) return;
    if (idx < 0 || idx >= smoothedContourSlicesData.length) return;

    const N = smoothedContourSlicesData.length;
    const M = 32;
    const scale = 0.07;
    const ySpacing_render = 1.0;
    const slices_count = 5;

    const slice = smoothedContourSlicesData[idx];
    const area = sliceAreasData[idx];

    // Find the max slice (maximum open airway contour) for point-by-point ratio calculations
    const maxIdx = sliceAreasData.indexOf(Math.max(...sliceAreasData));
    if (maxIdx < 0 || maxIdx >= smoothedContourSlicesData.length) return;
    const maxSlice = smoothedContourSlicesData[maxIdx];

    // Calculate centroid of current slice
    let cx = 0, cz = 0;
    for (let j = 0; j < M; j++) {
        cx += slice[j][0];
        cz += slice[j][1];
    }
    cx /= M;
    cz /= M;

    // Calculate centroid of maxSlice
    let maxCx = 0, maxCz = 0;
    for (let j = 0; j < M; j++) {
        maxCx += maxSlice[j][0];
        maxCz += maxSlice[j][1];
    }
    maxCx /= M;
    maxCz /= M;

    // Update airwayMesh geometry (dynamic morphing with point-by-point collapse ratio colors)
    if (airwayMesh && airwayMesh.geometry) {
        const posAttr = airwayMesh.geometry.getAttribute('position');
        const colorAttr = airwayMesh.geometry.getAttribute('color');

        for (let i = 0; i < slices_count; i++) {
            const zHeight = (i - Math.floor(slices_count / 2)) * ySpacing_render;
            for (let j = 0; j < M; j++) {
                const pt = slice[j];
                const x = (pt[0] - cx) * scale;
                const y = -(pt[1] - cz) * scale;

                // Point-by-point local collapse ratio
                const ptMax = maxSlice[j];
                const maxRadius = Math.sqrt((ptMax[0] - maxCx) ** 2 + (ptMax[1] - maxCz) ** 2);
                const currentRadius = Math.sqrt((pt[0] - cx) ** 2 + (pt[1] - cz) ** 2);
                const ratio = maxRadius > 0 ? (currentRadius / maxRadius) : 1.0;
                const clampedRatio = Math.max(0.0, Math.min(1.0, ratio));
                const color = getColorForRatio(clampedRatio);

                const vertexIndex = i * M + j;
                posAttr.setXYZ(vertexIndex, x, y, zHeight);
                colorAttr.setXYZ(vertexIndex, color.r, color.g, color.b);
            }
        }
        posAttr.needsUpdate = true;
        colorAttr.needsUpdate = true;
        airwayMesh.geometry.computeVertexNormals();
    }

    // Create glowing circle outline staying at z = 0.05
    const linePoints = [];
    const zHeightLine = 0.05;
    for (let j = 0; j < M; j++) {
        const pt = slice[j];
        const x = (pt[0] - cx) * scale;
        const y = -(pt[1] - cz) * scale;
        linePoints.push(new THREE.Vector3(x, y, zHeightLine));
    }
    linePoints.push(linePoints[0].clone()); // Close loop

    // Remove old indicator
    if (highlightLine) {
        scene.remove(highlightLine);
        highlightLine.geometry.dispose();
    }

    // Create glowing circle outline
    const lineGeom = new THREE.BufferGeometry().setFromPoints(linePoints);
    const lineMat = new THREE.LineBasicMaterial({
        color: 0x00f5ff, // Glowing cyan/neon blue
        linewidth: 4
    });

    highlightLine = new THREE.Line(lineGeom, lineMat);
    scene.add(highlightLine);

    // Clear old collapse arrow helpers
    arrowHelpers.forEach(arrow => scene.remove(arrow));
    arrowHelpers = [];

    // Add 4 collapse direction force vectors (Top, Bottom, Left, Right) at z = 0.0
    // 4 directions: index 0 (Right/Lateral R), 8 (Bottom/AP Posterior), 16 (Left/Lateral L), 24 (Top/AP Anterior)
    const targetIndices = [0, 8, 16, 24];
    targetIndices.forEach(j => {
        const ptMax = maxSlice[j];
        const ptCurr = slice[j];

        const xMax = (ptMax[0] - maxCx) * scale;
        const yMax = -(ptMax[1] - maxCz) * scale;
        const xCurr = (ptCurr[0] - cx) * scale;
        const yCurr = -(ptCurr[1] - cz) * scale;

        const startPt = new THREE.Vector3(xMax, yMax, 0.0);
        const endPt = new THREE.Vector3(xCurr, yCurr, 0.0);

        const dirVec = new THREE.Vector3().subVectors(endPt, startPt);
        const dist = dirVec.length();

        // Only draw arrow if displacement is noticeable
        if (dist > 0.03) {
            dirVec.normalize();
            const arrow = new THREE.ArrowHelper(
                dirVec,
                startPt,
                dist,
                0xef4444, // Red arrow representing collapse force
                0.35, // headLength
                0.2 // headWidth
            );
            scene.add(arrow);
            arrowHelpers.push(arrow);
        }
    });

    // Update overlay text in HTML 3D viewport
    const sliceInfo = document.getElementById('three-slice-info');
    if (sliceInfo) {
        sliceInfo.textContent = `Frame: ${idx + 1} / ${N} | Area: ${area.toFixed(0)} px²`;
    }
}

function startProcessing() {
    backendResultData = null; 
    const file = fileInput.files[0];
    if(!file) return;

    uploadSection.classList.remove('active');
    setTimeout(() => {
        uploadSection.style.display = 'none';
        processingSection.style.display = 'block';
        setTimeout(() => processingSection.classList.add('active'), 50);
        
        simulateSteps();
        
        const formData = new FormData();
        formData.append('video', file);
        const startTime = Date.now();
        const timeoutThreshold = 90000; // 90 seconds in milliseconds
        
        const uploadUrl = window.location.protocol === 'file:' ? 'http://127.0.0.1:5000/upload' : '/upload';
        console.log("[FRONTEND FETCH] Target Upload URL: ", uploadUrl);
        console.log("[FRONTEND FETCH] File details: Name=", file.name, " Size=", file.size, " bytes");

        // Set client-side timeout checking interval
        const timeoutChecker = setInterval(() => {
            const elapsed = Date.now() - startTime;
            if (elapsed >= timeoutThreshold && backendResultData === null) {
                console.error("[FRONTEND TIMEOUT] Request exceeded 90s timeout.");
                clearInterval(timeoutChecker);
                alert("Processing is taking too long. Please try uploading a shorter or smaller video resolution.");
                resetApp();
            }
        }, 1000);

        console.log("[FRONTEND LOG] Upload started. Waiting for backend response...");
        fetch(uploadUrl, { method: 'POST', body: formData })
        .then(async response => {
            console.log("[FRONTEND LOG] Response received from backend. Status: ", response.status);
            clearInterval(timeoutChecker); // Clear checker if response arrived
            
            if (!response.ok) {
                let errorMsg = `Server returned status ${response.status}: ${response.statusText}`;
                try {
                    const text = await response.text();
                    console.error("[FRONTEND FETCH] Error body text: ", text);
                    try {
                        const parsed = JSON.parse(text);
                        if (parsed && parsed.error) {
                            errorMsg = parsed.error;
                        }
                    } catch(e) {}
                } catch(e) {}
                throw new Error(errorMsg);
            }
            return response.json();
        })
        .then(data => {
            console.log("[FRONTEND LOG] JSON parsed successfully from response.");
            if(data.error) {
                alert("เกิดข้อผิดพลาดจากระบบประมวลผล: " + data.error);
                resetApp();
            } else {
                backendResultData = data;
                console.log("[FRONTEND LOG] Result data stored in backendResultData.");
            }
        })
        .catch(error => {
            clearInterval(timeoutChecker); // Clear checker on failure
            console.error("[FRONTEND FETCH] Critical failure: ", error);
            const detail = error.message || error.toString();
            let userFriendlyMsg = `ไม่สามารถติดต่อเครื่องเซิร์ฟเวอร์ประมวลผลภาพทางเรขาคณิตได้:\n\nรายละเอียดข้อผิดพลาด:\n${detail}\n\nข้อแนะนำสำหรับการแก้ปัญหา:\n`;
            if (detail.includes("Failed to fetch") || detail.includes("NetworkError") || detail.includes("status 504") || detail.includes("status 502")) {
                userFriendlyMsg += `1. เครื่องประมวลผลปลายทาง (Render Free tier) อาจกำลังหลับหรือค้างเนื่องจากหมดเวลาจำกัดของเซิร์ฟเวอร์ (Server Timeout/Restart)\n2. กรุณาทดลองส่งอัปโหลดไฟล์วิดีโอที่มีขนาดสั้นลง หรือขนาดความละเอียดภาพน้อยลง\n3. ตรวจสอบสถานะการเชื่อมต่ออินเทอร์เน็ตของคุณอีกครั้ง`;
            } else {
                userFriendlyMsg += `1. หากไฟล์มีขนาดใหญ่เกินพิกัด ระบบอาจปฏิเสธการอัปโหลด\n2. กรุณาลองอัปโหลดไฟล์วิดีโอที่มีขนาดความละเอียดน้อยลง หรือมีคาบช่วงเวลาที่สั้นลง\n3. รายละเอียดข้อผิดพลาดทางเทคนิค: ${detail}`;
            }
            alert(userFriendlyMsg);
            resetApp();
        });

    }, 500);
}

function simulateSteps() {
    let currentStep = 0;
    const interval = setInterval(() => {
        const progress = ((currentStep + 1) / steps.length) * 100;
        progressBar.style.width = `${progress}%`;
        if (currentStep > 0) {
            steps[currentStep - 1].classList.remove('active');
            steps[currentStep - 1].classList.add('completed');
        }
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(interval);
            checkIfAiIsDone();
        }
    }, 1000); 
}

function checkIfAiIsDone() {
    if(backendResultData === null) {
        setTimeout(checkIfAiIsDone, 500);
    } else {
        showResults();
    }
}

let animationInterval = null;

function showResults() {
    console.log("[FRONTEND LOG] Rendering results started...");
    try {
        processingSection.classList.remove('active');
        
        // Clear old looping animation
        if(animationInterval) clearInterval(animationInterval);
    
    // Load and build 3D geometry
    contourSlicesData = backendResultData.contour_slices || [];
    sliceAreasData = backendResultData.slice_areas || [];
    
    let maxIdx = 0;
    let minIdx = 0;
    
    if (contourSlicesData.length > 0) {
        render3DAirway(contourSlicesData, sliceAreasData);
        // Reset camera focus target to origin (Default perspective matching 2D view)
        camera.position.set(0, 0, 18);
        controls.target.set(0, 0, 0);
        controls.update();

        // วาดแผงเปรียบเทียบรูปทรงช่องลม 3 ระยะ (Max, Mid, Min)
        const maxAreaVal = Math.max(...sliceAreasData);
        const minAreaVal = Math.min(...sliceAreasData);
        maxIdx = sliceAreasData.indexOf(maxAreaVal);
        minIdx = sliceAreasData.indexOf(minAreaVal);
        const midIdx = Math.floor((maxIdx + minIdx) / 2);
        const midAreaVal = sliceAreasData[midIdx];

        drawPhaseContour('canvas-phase-max', contourSlicesData[maxIdx], maxAreaVal, maxAreaVal, '#10b981');
        drawPhaseContour('canvas-phase-mid', contourSlicesData[midIdx], midAreaVal, maxAreaVal, '#f59e0b');
        drawPhaseContour('canvas-phase-min', contourSlicesData[minIdx], minAreaVal, maxAreaVal, '#ef4444');

        document.getElementById('phase-area-max').textContent = maxAreaVal.toFixed(0) + ' px²';
        document.getElementById('phase-area-mid').textContent = midAreaVal.toFixed(0) + ' px²';
        document.getElementById('phase-area-min').textContent = minAreaVal.toFixed(0) + ' px²';
    }
    
    // Load images (base image is Max Open, overlay image is Max Collapse)
    const frames = backendResultData.sequence_frames || [];
    const maxOpenFrameBase64 = frames.length > 0 ? frames[maxIdx] : backendResultData.image_base64;
    const maxCollapseFrameBase64 = frames.length > 0 ? frames[minIdx] : backendResultData.image_base64;

    document.getElementById('result-img').src = maxOpenFrameBase64;
    if (collapseImg) {
        collapseImg.src = maxCollapseFrameBase64;
    }
    document.getElementById('heatmap-img').src = backendResultData.heatmap_base64;
    document.getElementById('contour-img').src = backendResultData.contour_base64;
    
    // Setup play/pause sync timeline
    isPlaying3D = true;
    currentPlaybackFrame = 0;
    const btnPlayPause = document.getElementById('btn-play-pause');
    if (btnPlayPause) {
        btnPlayPause.textContent = 'Pause';
        btnPlayPause.style.background = 'var(--primary)';
    }

    const timeline = document.getElementById('timeline-slider');
    if (timeline && frames.length > 0) {
        timeline.max = frames.length - 1;
        timeline.value = 0;
    }

    if (frames.length > 0) {
        updateUIForFrame(0);
        startPlaybackLoop();
    }
    
    // Reset Compare Slider position
    slider.value = 50;
    if (collapseImg) {
        collapseImg.style.clipPath = `polygon(50% 0, 100% 0, 100% 100%, 50% 100%)`;
    }
    heatmapImg.style.clipPath = `polygon(50% 0, 100% 0, 100% 100%, 50% 100%)`;
    contourImg.style.clipPath = `polygon(50% 0, 100% 0, 100% 100%, 50% 100%)`;
    sliderLine.style.left = `50%`;
    toggleHeatmap.checked = true;
    toggleContour.checked = true;
    heatmapImg.style.opacity = '1';
    contourImg.style.opacity = '1';

    // Set reduction percentage
    const reduction = backendResultData.reduction_percent;
    document.getElementById('val-reduction').textContent = reduction.toFixed(1) + '%';
    
    // Determine severity from VOTE classification values
    const severityBadge = document.getElementById('val-severity');
    const degree = backendResultData.degree;
    if (degree === 2) {
        severityBadge.innerHTML = `Severe Collapse (Degree 2: >75%)`;
        severityBadge.style.background = "#fef2f2";
        severityBadge.style.color = "#dc2626";
    } else if (degree === 1) {
        severityBadge.innerHTML = `Partial Collapse (Degree 1: 50-75%)`;
        severityBadge.style.background = "#fffbeb";
        severityBadge.style.color = "#d97706";
    } else {
        severityBadge.innerHTML = `Mild / Normal (Degree 0: <=50%)`;
        severityBadge.style.background = "#ecfdf5";
        severityBadge.style.color = "#059669";
    }

    // Set min lumen area text to the unified collapse_area_used_for_reduction
    document.getElementById('val-lumen').textContent = backendResultData.collapse_area_used_for_reduction.toFixed(0) + ' px²';

    // Set Automated Reference Metrics card text to unified values
    document.getElementById('val-reduction-report').textContent = reduction.toFixed(1) + '%';
    document.getElementById('val-lumen-report').textContent = backendResultData.collapse_area_used_for_reduction.toFixed(0) + ' px²';
    
    // Set max lumen area in the report
    const valMaxAreaReport = document.getElementById('val-max-area-report');
    if (valMaxAreaReport && backendResultData.max_lumen_area) {
        valMaxAreaReport.textContent = backendResultData.max_lumen_area.toFixed(0) + ' px²';
    }
    
    // Set breathing cycle frames using the refined actual open / collapse frame numbers
    const valPeakOpen = document.getElementById('val-peak-open');
    const valPeakCollapse = document.getElementById('val-peak-collapse');
    if (valPeakOpen && valPeakCollapse) {
        valPeakOpen.textContent = backendResultData.actual_open_frame + 1;
        valPeakCollapse.textContent = backendResultData.actual_collapse_frame + 1;
    }

    // Update Text Data for Geometry on screen
    const confVal = backendResultData.confidence;
    document.getElementById('geom-collapse-type').textContent = backendResultData.prediction_class + ' (Degree ' + backendResultData.degree + ')';
    document.getElementById('geom-confidence-value').textContent = 'Analysis Quality Score: ' + confVal.toFixed(1) + '%';
    document.getElementById('geom-progress-fill').style.width = confVal.toFixed(1) + '%';
    document.getElementById('clinical-reasoning').textContent = backendResultData.reasoning_text;

    // Expose Diagnostic Geometry Values to panel
    document.getElementById('dbg-frames').textContent = `${backendResultData.selected_cycle_open_frame} / ${backendResultData.selected_cycle_collapse_frame}`;
    document.getElementById('dbg-actual-frames').textContent = `${backendResultData.actual_open_frame} / ${backendResultData.actual_collapse_frame}`;
    document.getElementById('dbg-areas').textContent = `${backendResultData.max_lumen_area.toFixed(0)} / ${backendResultData.min_lumen_area.toFixed(0)} px²`;
    document.getElementById('dbg-collapse-used').textContent = `${backendResultData.collapse_area_used_for_reduction.toFixed(0)} px²`;
    document.getElementById('dbg-reduction').textContent = `${reduction.toFixed(1)}%`;
    document.getElementById('dbg-contours').textContent = `${backendResultData.valid_contour_count} / ${backendResultData.fallback_contour_count}`;
    document.getElementById('dbg-contours-lowest').textContent = backendResultData.fallback_count_in_lowest_10_percent;
    document.getElementById('dbg-aspect').textContent = backendResultData.avg_aspect.toFixed(2);
    document.getElementById('dbg-angle').textContent = backendResultData.avg_angle.toFixed(1) + '°';
    document.getElementById('dbg-bbox').textContent = `${(backendResultData.red_major * 100).toFixed(1)}% / ${(backendResultData.red_minor * 100).toFixed(1)}%`;
    
    // Injecting the new breathing cycle debug fields
    document.getElementById('dbg-num-cycles').textContent = backendResultData.num_cycles_evaluated;
    document.getElementById('dbg-cycle-score').textContent = `${backendResultData.selected_cycle_score.toFixed(1)} / ${backendResultData.selected_cycle_reduction.toFixed(1)}%`;
    document.getElementById('dbg-window-range').textContent = `${backendResultData.final_window_start + 1} - ${backendResultData.final_window_end}`;
    document.getElementById('dbg-window-enclose').textContent = `${backendResultData.final_window_contains_open ? 'YES' : 'NO'} / ${backendResultData.final_window_contains_collapse ? 'YES' : 'NO'}`;

    // Process optional Ground Truth Validation Mode comparisons
    const expClass = document.getElementById('expected-class').value;
    const expDegree = document.getElementById('expected-degree').value;
    const dbgAlert = document.getElementById('debug-validation-alert');
    
    if (expClass || expDegree !== "") {
        dbgAlert.style.display = 'block';
        let matchClass = true;
        let matchDegree = true;
        let alertHTML = "";

        if (expClass && expClass !== backendResultData.prediction_class) matchClass = false;
        if (expDegree !== "" && parseInt(expDegree) !== backendResultData.degree) matchDegree = false;

        if (matchClass && matchDegree) {
            dbgAlert.style.background = "#dcfce7";
            dbgAlert.style.color = "#15803d";
            dbgAlert.style.border = "1px solid #bbf7d0";
            alertHTML = `✅ VALIDATION PASSED (Matches expected ${expClass || 'any'} D${expDegree !== "" ? expDegree : 'any'})`;
        } else {
            dbgAlert.style.background = "#fee2e2";
            dbgAlert.style.color = "#b91c1c";
            dbgAlert.style.border = "1px solid #fecaca";
            alertHTML = `❌ MISMATCH DETECTED<br>Expected: ${expClass || 'N/A'} (Degree ${expDegree !== "" ? expDegree : 'N/A'})<br>Actual: ${backendResultData.prediction_class} (Degree ${backendResultData.degree})`;
        }
        dbgAlert.innerHTML = alertHTML;
    } else {
        dbgAlert.style.display = 'none';
    }

    // Dynamic warning injection based on diagnostics values
    const warningBox = document.getElementById('borderline-warning');
    const warningList = document.getElementById('warning-list');
    if (warningBox && warningList) {
        let warnings = backendResultData.warnings || [];
        
        if (backendResultData.close_to_threshold) {
            warnings.push("ค่าอัตราการยุบตัวของพื้นที่ทางเดินหายใจใกล้เคียงเกณฑ์เปลี่ยนระดับความรุนแรง (Borderline Area Reduction threshold)");
        }
        if (backendResultData.fallback_contour_count > (backendResultData.valid_contour_count + backendResultData.fallback_contour_count) * 0.3) {
            warnings.push("ตรวจพบอัตราการใช้เส้นสมมติช่วยชดเชยระดับสูงเนื่องจากขอบผนังกลืนไปกับทางเดินหายใจ (High Fallback Contour Rate)");
        }
        if (backendResultData.min_lumen_area < 50.0) {
            warnings.push("ขอบเขตช่องคอแฟบลงจนปิดสนิทอย่างสมบูรณ์ (Lumen min area near zero)");
        }
        
        if (backendResultData.is_low_reliability || warnings.length > 0) {
            warningBox.style.display = 'block';
            warningList.innerHTML = '';
            warnings.forEach(w => {
                const li = document.createElement('li');
                li.textContent = w;
                warningList.appendChild(li);
            });
        } else {
            warningBox.style.display = 'none';
        }
    }

    // Handle Clinical Reference Gold Standard Comparison Card inside Debug panel
    const refBlock = document.getElementById('dbg-clinical-ref-block');
    if (backendResultData.clinical_reference && refBlock) {
        const ref = backendResultData.clinical_reference;
        refBlock.style.display = 'block';
        
        document.getElementById('ref-physician-class').textContent = ref.class;
        document.getElementById('ref-physician-degree').textContent = 'Degree ' + ref.degree;
        document.getElementById('ref-physician-reduction').textContent = ref.reduction.toFixed(1) + '%';
        document.getElementById('ref-physician-reasoning').innerHTML = ref.reasoning.replace(/\n/g, '<br>');
    } else {
        if (refBlock) refBlock.style.display = 'none';
    }

    // Synchronize Printable Report Fields
    document.getElementById('print-val-reduction').textContent = reduction.toFixed(1) + '%';
    document.getElementById('print-val-lumen').textContent = backendResultData.collapse_area_used_for_reduction.toFixed(0) + ' px²';
    document.getElementById('print-geom-type').textContent = backendResultData.prediction_class + ' (Degree ' + backendResultData.degree + ')';
    
    let severityText = "";
    if (degree === 2) {
        severityText = "Severe Collapse (ระดับ 2: >75%)";
    } else if (degree === 1) {
        severityText = "Partial Collapse (ระดับ 1: 50-75%)";
    } else {
        severityText = "Mild / Normal (ระดับ 0: <=50%)";
    }
    document.getElementById('print-val-severity').textContent = severityText;

    setTimeout(() => {
        processingSection.style.display = 'none';
        resultSection.style.display = 'block';
        setTimeout(() => {
            resultSection.classList.add('active');
            onWindowResize(); // Force WebGL viewport adjustment once visible
        }, 50);
    }, 500);

    console.log("[FRONTEND LOG] Rendering completed successfully.");
    } catch (renderError) {
        console.error("[FRONTEND LOG] Exception occurred during rendering: ", renderError);
        alert("ข้อผิดพลาดฝั่งไคลเอนต์ขณะแสดงผลข้อมูลเรขาคณิต (Rendering Error):\n\n" + (renderError.stack || renderError.message || renderError));
        resetApp();
    }
}

function resetApp() {
    resultSection.classList.remove('active');
    
    // Reset active tab to Overview tab
    switchTab('tab-overview');
    
    // Hide AI card on reset
    const aiCard = document.getElementById('ai-result-card');
    if (aiCard) aiCard.style.display = 'none';

    // Clear 3-Phase canvases and text on reset
    ['max', 'mid', 'min'].forEach(phase => {
        const canvas = document.getElementById(`canvas-phase-${phase}`);
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        const text = document.getElementById(`phase-area-${phase}`);
        if (text) text.textContent = '-- px²';
    });
    
    // Clear breathing signal canvas on reset
    const signalCanvas = document.getElementById('canvas-breathing-signal');
    if (signalCanvas) {
        const ctx = signalCanvas.getContext('2d');
        ctx.clearRect(0, 0, signalCanvas.width, signalCanvas.height);
    }
    const signalFrameText = document.getElementById('signal-active-frame');
    if (signalFrameText) {
        signalFrameText.textContent = '1';
    }
    
    // Clear playback intervals
    if (playbackInterval) {
        clearInterval(playbackInterval);
        playbackInterval = null;
    }
    isPlaying3D = true;
    currentPlaybackFrame = 0;
    
    // Clear 3D airway objects from scene
    if (airwayMesh) {
        scene.remove(airwayMesh);
        airwayMesh.geometry.dispose();
        airwayMesh.material.dispose();
        airwayMesh = null;
    }
    if (referenceMesh) {
        scene.remove(referenceMesh);
        referenceMesh.geometry.dispose();
        referenceMesh.material.dispose();
        referenceMesh = null;
    }
    if (highlightLine) {
        scene.remove(highlightLine);
        highlightLine.geometry.dispose();
        highlightLine.material.dispose();
        highlightLine = null;
    }
    arrowHelpers.forEach(arrow => scene.remove(arrow));
    arrowHelpers = [];

    contourSlicesData = [];
    sliceAreasData = [];
    const sliceInfo = document.getElementById('three-slice-info');
    if (sliceInfo) sliceInfo.textContent = "Slice: -";
    
    setTimeout(() => {
        resultSection.style.display = 'none';
        progressBar.style.width = '0%';
        steps.forEach(step => step.classList.remove('active', 'completed'));
        steps[0].classList.add('active');
        fileInput.value = '';
        
        uploadSection.style.display = 'block';
        setTimeout(() => uploadSection.classList.add('active'), 50);
    }, 500);
}

// Function to draw vector contour shapes on Canvas for Max, Mid, Min phases
function drawPhaseContour(canvasId, points, area, maxArea, colorHex) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    
    // Background
    ctx.fillStyle = '#ffffff'; // Match dashboard background (light theme)
    ctx.fillRect(0, 0, w, h);
    
    // Draw visual helper grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 25; x < w; x += 25) {
        ctx.moveTo(x, 0); ctx.lineTo(x, h);
        ctx.moveTo(0, x); ctx.lineTo(w, x);
    }
    ctx.stroke();
    
    if (!points || points.length === 0) return;
    
    // Base scaling to map 224x224 coordinates to canvas.
    // We want the relative sizes of all three canvases to match their actual area proportions,
    // so we scale all of them using the same base factor determined by the Max Open (maxArea) bounding size.
    const baseScale = (w * 0.85) / 224;
    
    ctx.beginPath();
    for (let j = 0; j < points.length; j++) {
        const pt = points[j];
        // Translate center (112, 112) of 224x224 frame to canvas center (w/2, h/2)
        const cx = w / 2 + (pt[0] - 112) * baseScale;
        const cy = h / 2 + (pt[1] - 112) * baseScale;
        if (j === 0) {
            ctx.moveTo(cx, cy);
        } else {
            ctx.lineTo(cx, cy);
        }
    }
    ctx.closePath();
    
    // Shape fill (semi-transparent)
    ctx.fillStyle = colorHex + '18'; // ~10% opacity
    ctx.fill();
    
    // Shape stroke
    ctx.strokeStyle = colorHex;
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Center crosshair marker
    ctx.fillStyle = 'rgba(15, 23, 42, 0.15)';
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 2.5, 0, 2 * Math.PI);
    ctx.fill();
}

// Function to manage automatic animation interval scheduling
function startPlaybackLoop() {
    if (playbackInterval) clearInterval(playbackInterval);
    
    playbackInterval = setInterval(() => {
        if (!isPlaying3D) return;
        
        const frames = backendResultData ? (backendResultData.sequence_frames || []) : [];
        if (frames.length === 0) return;
        
        currentPlaybackFrame = (currentPlaybackFrame + 1) % frames.length;
        updateUIForFrame(currentPlaybackFrame);
    }, playbackSpeedMs);
}

// Function to synchronize all visual elements to a specific frame index
function updateUIForFrame(frameIdx) {
    currentPlaybackFrame = frameIdx;
    
    // Sync 2D animation frame
    const gifImg = document.getElementById('gif-animation');
    const frames = backendResultData ? (backendResultData.sequence_frames || []) : [];
    if (frames.length > 0 && gifImg) {
        gifImg.src = frames[frameIdx];
    }
    
    // Sync 3D level plane and collapse arrows
    updateHighlightSlice(frameIdx);
    
    // Sync Timeline Slider
    const timeline = document.getElementById('timeline-slider');
    if (timeline) {
        timeline.value = frameIdx;
    }
    
    // Sync 2D Breathing Wave Signal Graph
    drawBreathingSignal(frameIdx);
}

// Function to draw dynamic high-DPI respiratory breathing wave graph
function drawBreathingSignal(frameIdx) {
    const canvas = document.getElementById('canvas-breathing-signal');
    if (!canvas) return;
    
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);
    
    if (!sliceAreasData || sliceAreasData.length === 0) return;
    
    const N = sliceAreasData.length;
    const maxArea = Math.max(...sliceAreasData);
    const minArea = Math.min(...sliceAreasData);
    const range = maxArea - minArea;
    
    // Draw background horizontal grid lines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    for (let yVal = 20; yVal < h; yVal += 25) {
        ctx.beginPath();
        ctx.moveTo(0, yVal);
        ctx.lineTo(w, yVal);
        ctx.stroke();
    }
    
    // Coordinates mapping
    const getX = (idx) => (idx / (N - 1)) * (w - 40) + 20;
    const getY = (area) => {
        if (maxArea === minArea) return h / 2;
        const norm = (area - minArea) / range;
        return h - (norm * (h - 35) + 15);
    };
    
    // 1. Draw area under curve (Smooth Linear Gradient Fill)
    ctx.beginPath();
    ctx.moveTo(getX(0), h);
    for (let i = 0; i < N; i++) {
        ctx.lineTo(getX(i), getY(sliceAreasData[i]));
    }
    ctx.lineTo(getX(N - 1), h);
    ctx.closePath();
    
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(2, 132, 199, 0.16)');
    grad.addColorStop(1, 'rgba(2, 132, 199, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();
    
    // 2. Draw graph line
    ctx.beginPath();
    ctx.moveTo(getX(0), getY(sliceAreasData[0]));
    for (let i = 1; i < N; i++) {
        ctx.lineTo(getX(i), getY(sliceAreasData[i]));
    }
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    
    // 3. Draw vertical sync cursor line
    const curX = getX(frameIdx);
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(curX, 5);
    ctx.lineTo(curX, h - 5);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 4. Draw active point (Glow circle)
    const curY = getY(sliceAreasData[frameIdx]);
    ctx.beginPath();
    ctx.arc(curX, curY, 6, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.25)';
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(curX, curY, 3.5, 0, 2 * Math.PI);
    ctx.fillStyle = '#ef4444';
    ctx.fill();
    
    // 5. Update numerical overlay text
    const activeLabel = document.getElementById('signal-active-frame');
    if (activeLabel) {
        activeLabel.textContent = frameIdx + 1;
    }
}

// Function to handle tab switching inside results dashboard
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
        }
    });
    
    const targetContent = document.getElementById(tabId);
    if (targetContent) {
        targetContent.classList.add('active');
    }
    
    // Trigger window resize to recalculate Three.js viewport
    if (tabId === 'tab-overview') {
        setTimeout(onWindowResize, 50);
    }
}

// Removed physician manual override logic

// Function to handle printing of diagnostic reports
function printReport() {
    if (!backendResultData) {
        alert("กรุณารอให้ระบบประมวลผลวิดีโอเสร็จสิ้นก่อนสั่งพิมพ์รายงาน");
        return;
    }
    
    // Sync screenshot images
    const contourImg = document.getElementById('contour-img');
    const printContour = document.getElementById('print-contour-img');
    if (contourImg && printContour) {
        printContour.src = contourImg.src;
    }
    
    const heatmapImg = document.getElementById('heatmap-img');
    const printHeatmap = document.getElementById('print-heatmap-img');
    if (heatmapImg && printHeatmap) {
        printHeatmap.src = heatmapImg.src;
    }
    
    // Sync print date
    const printDate = document.getElementById('print-report-date');
    if (printDate) {
        printDate.textContent = new Date().toLocaleDateString('th-TH', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }
    
    // Open system print dialog
    window.print();
}
