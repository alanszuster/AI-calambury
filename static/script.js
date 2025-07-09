let canvas, ctx;
let isDrawing = false;
let currentWord = '';

document.addEventListener('DOMContentLoaded', function() {
    initCanvas();
    setupEventListeners();
    showWelcomeMessage();
});

function initCanvas() {
    canvas = document.getElementById('drawing-canvas');
    ctx = canvas.getContext('2d');

    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    clearCanvas();
}

function setupEventListeners() {
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);

    document.getElementById('clear-btn').addEventListener('click', clearCanvas);
    document.getElementById('predict-btn').addEventListener('click', predictDrawing);
    document.getElementById('new-word-btn').addEventListener('click', getNewWord);
}

function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(
        e.type === 'touchstart' ? 'mousedown' :
        e.type === 'touchmove' ? 'mousemove' : 'mouseup',
        {
            clientX: touch.clientX,
            clientY: touch.clientY
        }
    );
    canvas.dispatchEvent(mouseEvent);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
}

async function getNewWord() {
    try {
        showLoadingMessage('Choosing new word...');

        const response = await fetch('/get_random_word');
        const data = await response.json();

        currentWord = data.word;
        document.getElementById('current-word').textContent = currentWord;

        clearCanvas();
        showWelcomeMessage();

        const wordElement = document.getElementById('current-word');
        wordElement.classList.add('success');
        setTimeout(() => wordElement.classList.remove('success'), 600);

    } catch (error) {
        console.error('Error getting new word:', error);
        showError('Cannot get new word. Please try again.');
    }
}

async function predictDrawing() {
    try {
        showLoadingMessage('AI is analyzing your drawing...');

        const imageData = canvas.toDataURL('image/png');

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.success && data.predictions.length > 0) {
            displayPredictions(data.predictions);
        } else {
            showError(data.error || 'Cannot recognize the drawing. Try drawing more clearly.');
        }

    } catch (error) {
        console.error('Error during prediction:', error);
        showError('An error occurred during analysis. Please try again.');
    }
}

function displayPredictions(predictions) {
    const predictionsDiv = document.getElementById('predictions');

    let html = '<h4>🔍 What AI sees in your drawing:</h4>';

    predictions.forEach((pred, index) => {
        const isCorrect = pred.class.toLowerCase() === currentWord.toLowerCase();
        const emoji = index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉';
        const correctClass = isCorrect ? ' style="border-left-color: #28a745; background: #d4edda;"' : '';

        html += `
            <div class="prediction-item"${correctClass}>
                <div class="prediction-class">
                    ${emoji} ${pred.class}
                    ${isCorrect ? ' ✅ CORRECT!' : ''}
                </div>
                <div class="prediction-confidence">Confidence: ${pred.confidence}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${pred.confidence}%"></div>
                </div>
            </div>
        `;
    });

    if (predictions.some(p => p.class.toLowerCase() === currentWord.toLowerCase())) {
        html += '<div style="text-align: center; margin-top: 20px; font-size: 1.2em; color: #28a745;">🎉 Congratulations! AI guessed your drawing!</div>';
    }

    predictionsDiv.innerHTML = html;
}

function showLoadingMessage(message) {
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = `
        <div class="loading" style="text-align: center; padding: 20px;">
            <div style="font-size: 2em; margin-bottom: 10px;">🤖</div>
            <div>${message}</div>
        </div>
    `;
}

function showWelcomeMessage() {
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = `
        <div style="text-align: center; padding: 20px; color: #666;">
            <div style="font-size: 3em; margin-bottom: 15px;">🎨</div>
            <p>Draw something on the canvas and I'll show you what I see!</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Tip: Draw clearly and use the entire canvas area</p>
        </div>
    `;
}

function showError(message) {
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = `
        <div style="text-align: center; padding: 20px; color: #dc3545;">
            <div style="font-size: 2em; margin-bottom: 10px;">❌</div>
            <div>${message}</div>
        </div>
    `;
}

function downloadDrawing() {
    const link = document.createElement('a');
    link.download = `ai-pictionary-${currentWord || 'drawing'}.png`;
    link.href = canvas.toDataURL();
    link.click();
}

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case 'n':
                e.preventDefault();
                getNewWord();
                break;
            case 'Enter':
                e.preventDefault();
                predictDrawing();
                break;
            case 'Backspace':
                e.preventDefault();
                clearCanvas();
                break;
        }
    }
});

console.log('🎨 AI Pictionary loaded! Have fun!');
