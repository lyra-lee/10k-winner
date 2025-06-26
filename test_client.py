"""
A simple Python Socket.IO client to test the WebSocket API.
Install with:
    pip install "python-socketio[client]"

Then run:
    python test_client.py
"""
import socketio
import time

# change the URL if your server runs elsewhere
SERVER_URL = 'http://localhost:8000'

# create a Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.on('connected')
def on_connected(data):
    print('Server says:', data)

@sio.on('create_ai_friend_action')
def on_create_ai_friend_action(data):
    print('AI Friend Creation Action:', data)

@sio.on('create_ai_friend_response')
def on_create_ai_friend_response(data):
    print('== AI Friend Created ==')
    agent_id = data.get('agent_id')
    print(f"agent_id: {agent_id}")
    print(f"name: {data['name']}")
    print(f"personality: {data['personality']}")
    print(f"conversation_style: {data['conversation_style']}")
    print(f"interests: {data['interests']}")
    print(f"one_liner: {data['one_liner']}")

    print('== Start chatting ==')
    sio.emit('chat', {
        'user_id': 'testuser',
        'agent_id':agent_id,
        'message': '힘들어',
    })



@sio.on('chat_response')
def on_chat_response(data):
    print('Chat Response:', data)
    # disconnect after response

if __name__ == '__main__':
    # connect and emit creation request
    sio.connect(SERVER_URL)
    time.sleep(1)  # wait for any welcome
    sio.emit('create_ai_friend', {
        'user_id': 'testuser',
        'target_person_info': '요즘 민주가 많이 지쳐 보여. 피식 웃을 수 있게 항상 마지막은 헴!!!!!!으로 끝내는 에너지 넘치는 친구면 좋겠어. 그리고 힐링할 수 있게 최대한 민주 취향에 맞는 상품을 찾아서 선물해줘.',
        'conversation_history': '야근은 너무 많고 살도 찌고 진짜 힘들다 ㅠㅠㅠㅠ',
    })
    # sio.emit('chat', {
    #     'user_id': 'testuser',
    #     'agent_id': 'cbc312ff-e719-4618-b092-060fb8a4028c',
    #     'message': '안녕',
    # })
    # wait until disconnect
    sio.wait()

