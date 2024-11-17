from flask import Flask, request, jsonify
from utilities.simple_rag import simple_rag_call
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        session_id = data.get('session_id', None)  # Optional session_id
        
        response, chat_history, new_session_id = simple_rag_call(query, session_id)
        
        return jsonify({
            'response': response,
            'session_id': new_session_id,
            'chat_history': [
                {
                    'role': message.type,
                    'content': message.content
                } for message in chat_history
            ]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 