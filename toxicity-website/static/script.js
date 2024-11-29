document.getElementById('toxicity-form').addEventListener('submit', function (event) {
    event.preventDefault();

    let comment = document.getElementById('comment').value;

    if (comment.trim() === '') {
        alert('Please enter a comment.');
        return;
    }

    // Disable button while waiting for response
    const button = event.target.querySelector('button');
    button.disabled = true;
    button.textContent = 'Analyzing...';

    // Send the comment to the Flask backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `comment=${encodeURIComponent(comment)}`
    })
    .then(response => response.json())
    .then(data => {
        // Enable the button again
        button.disabled = false;
        button.textContent = 'Analyze';

        // Display results
        let resultList = document.getElementById('result-list');
        resultList.innerHTML = '';

        for (const [label, result] of Object.entries(data)) {
            let div = document.createElement('div');
            div.classList.add('results-item');

            let labelSpan = document.createElement('span');
            labelSpan.classList.add('label');
            labelSpan.textContent = label.replace(/_/g, ' ').toUpperCase() + ':';

            let scoreSpan = document.createElement('span');
            scoreSpan.classList.add('score');
            scoreSpan.classList.add(result.label === 'Yes' ? 'positive' : 'negative');
            scoreSpan.textContent = `${result.label} (${result.score})`;

            div.appendChild(labelSpan);
            div.appendChild(scoreSpan);
            resultList.appendChild(div);
        }

        document.getElementById('results').style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('There was an error analyzing the comment.');
        button.disabled = false;
        button.textContent = 'Analyze';
    });
});
