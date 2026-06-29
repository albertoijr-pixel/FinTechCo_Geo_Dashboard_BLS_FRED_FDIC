import os
import json
import io
import pandas as pd
import anthropic
from flask import Flask, render_template, jsonify, request, Response
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'target_scores.csv')


def load_data():
    return pd.read_csv(DATA_PATH)


def apply_filters(df, min_score=0, states=None, sectors=None):
    if min_score:
        df = df[df['target_score'] >= float(min_score)]
    if states:
        upper = [s.upper() for s in states]
        df = df[df['state'].str.upper().isin(upper)]
    if sectors:
        df = df[df['dominant_sector'].isin(sectors)]
    return df


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    df = load_data()
    return jsonify(df.where(pd.notnull(df), None).to_dict(orient='records'))


@app.route('/api/filter', methods=['POST'])
def filter_data():
    body = request.json or {}
    df = load_data()
    df = apply_filters(
        df,
        min_score=body.get('min_score', 0),
        states=body.get('states', []),
        sectors=body.get('sectors', []),
    )
    return jsonify(df.where(pd.notnull(df), None).to_dict(orient='records'))


@app.route('/api/export', methods=['POST'])
def export_data():
    body = request.json or {}
    df = load_data()
    df = apply_filters(
        df,
        min_score=body.get('min_score', 0),
        states=body.get('states', []),
        sectors=body.get('sectors', []),
    )
    output = io.StringIO()
    df.to_csv(output, index=False)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=target_scores_filtered.csv'},
    )


@app.route('/api/chat', methods=['POST'])
def chat():
    body = request.json or {}
    message = body.get('message', '')
    context = body.get('context', '[]')

    system_prompt = (
        'You are a FinTech data analyst assistant helping a marketing team identify '
        'high-value geographic targets for a digital deposit and credit card campaign. '
        'The user is viewing a dashboard of US states scored by deposit concentration, '
        'non-interest-bearing deposits, and high-wage professional density (Tech, Healthcare, Legal). '
        'Answer questions concisely in plain business language. '
        f'Here is the current dataset: {context}\n\n'
        'IMPORTANT: If the user asks to filter, show, highlight, or focus on specific states or data subsets, '
        'include a filter_command in your JSON response in this format: '
        '{"filter_command": {"states": ["CALIFORNIA", "NEW YORK"], "min_score": 20, "sectors": ["Tech"]}}. '
        'If no filter is requested, return "filter_command": null. '
        'Always return valid JSON with both "response" and "filter_command" keys.'
    )

    text = ''
    try:
        ai_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        resp = ai_client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1024,
            system=system_prompt,
            messages=[{'role': 'user', 'content': message}],
        )
        text = resp.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith('```'):
            lines = text.split('\n')
            end = len(lines) - 1 if lines[-1].strip() == '```' else len(lines)
            text = '\n'.join(lines[1:end]).strip()
            if text.startswith('json'):
                text = text[4:].strip()
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {'response': text or 'Could not parse response.', 'filter_command': None}
    except Exception as exc:
        result = {'response': f'Error: {exc}', 'filter_command': None}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
