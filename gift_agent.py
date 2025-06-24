from flask import Flask, request, jsonify
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain import hub
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, ClassVar
import json
from openai import OpenAI
import os
from datetime import datetime

app = Flask(__name__)

# 환경 변수 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')

client = OpenAI(api_key=OPENAI_API_KEY)

class UserSession:
    def __init__(self, user_id: str, budget: float = 10000):
        self.user_id = user_id
        self.budget = budget
        self.spent_tokens = 0
        self.spent_gifts = 0
        self.conversation_history = []
        self.mood_analysis = {}

    def get_remaining_budget(self) -> float:
        return self.budget - self.spent_tokens - self.spent_gifts

    def can_afford(self, cost: float) -> bool:
        return self.get_remaining_budget() >= cost

class AIFriendProfile:
    """생성될 AI 친구의 프로필"""
    def __init__(self):
        self.agent_id = str(uuid.uuid4())
        self.name = ""
        self.personality = ""
        self.conversation_style = ""
        self.interests = []
        self.special_memories = []
        self.relationship_context = ""
        self.gift_preferences = {}
        self.conversation_starters = []
        self.created_at = datetime.now().isoformat()


# 전역 세션 저장소 (실제 환경에서는 Redis나 DB 사용)
user_sessions: Dict[str, UserSession] = {}
ai_friend_profiles: Dict[str, AIFriendProfile] = {}


class BudgetCalculatorTool(BaseTool):
    name: str = "budget_calculator"
    description: str = "예산을 확인하고 토큰 비용을 계산하는 도구입니다. 남은 예산을 확인하거나 특정 작업의 비용을 계산할 때 사용하세요."

    def _run(self, query: str, user_id: str = None) -> str:
        if user_id not in user_sessions:
            return "사용자 세션을 찾을 수 없습니다."

        session = user_sessions[user_id]

        if "잔액" in query or "남은" in query:
            return f"현재 잔액: {session.get_remaining_budget():.0f}원\n토큰 사용: {session.spent_tokens:.0f}원\n선물 비용: {session.spent_gifts:.0f}원"

        return f"총 예산: {session.budget}원, 남은 예산: {session.get_remaining_budget():.0f}원"

    def _arun(self, query: str, user_id: str = None):
        raise NotImplementedError("비동기 실행은 지원되지 않습니다.")


class MoodAnalyzerTool(BaseTool):
    name: str = "mood_analyzer"
    description: str = "사용자의 메시지를 분석하여 현재 기분과 감정 상태를 파악하는 도구입니다. 대화 내용을 바탕으로 사용자의 감정을 분석합니다."

    def _run(self, message: str, user_id: str = None) -> str:
        # OpenAI API를 사용하여 감정 분석
        try:
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "사용자의 메시지를 분석하여 감정 상태를 JSON 형태로 분석해주세요. 다음 형태로 응답하세요: {\"mood\": \"기분상태\", \"emotion_score\": 1-10점수, \"keywords\": [\"감정키워드들\"], \"recommended_action\": \"추천행동\"}"},
                {"role": "user", "content": message}
            ],
            max_tokens=200,
            temperature=0.3)

            analysis = response.choices[0].message.content

            if user_id and user_id in user_sessions:
                # 토큰 비용 계산 (대략적인 계산)
                tokens_used = response.usage.total_tokens
                cost = tokens_used * 0.002  # 대략적인 토큰당 비용
                user_sessions[user_id].spent_tokens += cost
                user_sessions[user_id].mood_analysis = json.loads(analysis)

            return f"감정 분석 결과: {analysis}"

        except Exception as e:
            return f"감정 분석 중 오류가 발생했습니다: {str(e)}"

    def _arun(self, message: str, user_id: str = None):
        raise NotImplementedError("비동기 실행은 지원되지 않습니다.")


class GiftSelectorTool(BaseTool):
    name: str = "gift_selector"
    description: str = "사용자의 기분과 예산에 맞는 선물을 선택하는 도구입니다. 감정 상태와 남은 예산을 고려하여 적절한 선물을 추천합니다."

    # ClassVar로 어노테이션하여 Pydantic 필드가 아님을 명시
    gift_catalog: ClassVar[Dict[str, Dict[str, Any]]] = {
        "스티커": {"price": 100, "mood": ["행복", "즐거움", "기쁨"]},
        "이모티콘": {"price": 200, "mood": ["귀여움", "사랑", "애정"]},
        "커피 쿠폰": {"price": 5000, "mood": ["피곤", "스트레스", "힘듦"]},
        "꽃다발": {"price": 15000, "mood": ["슬픔", "위로", "사랑"]},
        "초콜릿": {"price": 3000, "mood": ["달달함", "위로", "행복"]},
        "음악 선물": {"price": 1000, "mood": ["감성", "추억", "그리움"]},
    }

    def _run(self, mood_info: str, user_id: str = None) -> str:
        if user_id not in user_sessions:
            return "사용자 세션을 찾을 수 없습니다."

        session = user_sessions[user_id]
        remaining_budget = session.get_remaining_budget()

        # 예산 내에서 선물 필터링
        affordable_gifts = {name: info for name, info in self.gift_catalog.items()
                            if info["price"] <= remaining_budget}

        if not affordable_gifts:
            return "예산이 부족하여 선물을 준비할 수 없습니다. 따뜻한 말로 위로해드릴게요! 💝"

        # 기분에 맞는 선물 추천
        mood_keywords = session.mood_analysis.get("keywords", [])
        current_mood = session.mood_analysis.get("mood", "")

        best_gift = None
        best_score = 0

        for gift_name, gift_info in affordable_gifts.items():
            score = 0
            for keyword in mood_keywords:
                if any(mood in keyword.lower() for mood in gift_info["mood"]):
                    score += 1

            if score > best_score or (score == best_score and best_gift is None):
                best_gift = gift_name
                best_score = score

        # 기본 추천 (매칭되는 것이 없을 때)
        if best_gift is None:
            best_gift = min(affordable_gifts.keys(), key=lambda x: affordable_gifts[x]["price"])

        gift_price = self.gift_catalog[best_gift]["price"]
        session.spent_gifts += gift_price

        return f"🎁 선물 추천: {best_gift} (가격: {gift_price}원)\n남은 예산: {session.get_remaining_budget():.0f}원\n선물을 보냈습니다! 💝"

    def _arun(self, mood_info: str, user_id: str = None):
        raise NotImplementedError("비동기 실행은 지원되지 않습니다.")


class ConversationTool(BaseTool):
    name: str = "conversation_generator"
    description: str = "사용자의 기분에 맞는 대화를 생성하는 도구입니다. 감정 상태를 고려하여 적절한 응답을 만들어줍니다."

    def _run(self, user_message: str, user_id: str = None) -> str:
        if user_id not in user_sessions:
            return "안녕하세요! 오늘 기분이 어떠세요?"

        session = user_sessions[user_id]
        mood_info = session.mood_analysis

        # 기분에 따른 대화 톤 조절
        mood = mood_info.get("mood", "보통")
        emotion_score = mood_info.get("emotion_score", 5)

        try:
            # 예산 확인
            if not session.can_afford(50):  # 대화 최소 비용
                return "예산이 부족해서 더 이상 대화를 이어갈 수 없어요. 하지만 당신을 응원하고 있어요! 💕"

            system_message = f"""
            당신은 따뜻하고 친근한 AI 친구입니다. 사용자의 현재 기분은 '{mood}'이고, 감정 점수는 {emotion_score}/10입니다.
            이 정보를 바탕으로 공감하고 위로하며, 적절한 반응을 보여주세요.
            또한 당신은 사용자의 요구에 맞게 친구처럼 대화를 이어나갈 수 있어야 합니다.
            사용자에게 지금 맞는 행동이 무엇인지 파악하기 위한 질문을 계속 던지세요.
            
            중요: 대화를 이어가는 것에 중점을 두어 사용자의 응답을 유도해주세요. 먼저 질문을 하고, 사용자가 이에 응답할수 있게끔 대화를 이어가세요.
            
            답변은 2-3문장 내외로 자연스럽게 해주세요.
            """

            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7)

            # 토큰 비용 계산
            tokens_used = response.usage.total_tokens
            cost = tokens_used * 0.002
            session.spent_tokens += cost

            ai_response = response.choices[0].message.content

            # 대화 히스토리 저장
            session.conversation_history.append({
                "user": user_message,
                "ai": ai_response,
                "timestamp": datetime.now().isoformat(),
                "cost": cost
            })

            return ai_response

        except Exception as e:
            return f"대화 생성 중 오류가 발생했습니다: {str(e)}"

    def _arun(self, user_message: str, user_id: str = None):
        raise NotImplementedError("비동기 실행은 지원되지 않습니다.")

class UserIdToolWrapper(BaseTool):
    """
    기존 도구에 user_id를 추가로 전달하기 위한 래퍼 클래스.
    BaseTool을 상속받아 LangChain과 호환되도록 만듭니다.
    """
    name: str
    description: str
    tool: BaseTool  # 원본 도구 객체를 저장
    user_id: str

    def _run(self, query: str) -> str:
        """이 도구가 실행될 때 호출되는 동기 메서드"""
        # 원본 도구의 _run 메서드를 user_id와 함께 호출
        return self.tool._run(query, user_id=self.user_id)

    async def _arun(self, query: str) -> str:
        """비동기 실행이 필요할 경우를 위한 메서드"""
        # 원본 도구가 비동기 실행을 지원하는지 확인
        if hasattr(self.tool, '_arun'):
            return await self.tool._arun(query, user_id=self.user_id)
        # 지원하지 않으면 동기 메서드를 실행
        return self._run(query)

# LangChain Agent 설정
def create_agent(user_id: str):
    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )

    tools = [
        BudgetCalculatorTool(),
        MoodAnalyzerTool(),
        GiftSelectorTool(),
        ConversationTool()
    ]

    wrapped_tools = [
        UserIdToolWrapper(
            name=tool.name,
            description=tool.description,
            tool=tool,
            user_id=user_id
        ) for tool in tools
    ]

    # ReAct 프롬프트 템플릿 가져오기
    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template("""
     당신은 사용자의 감정을 이해하고 공감하는 따뜻한 AI 친구입니다.
    다음 도구들을 사용해서 사용자를 도와주세요:

    {tools}

    다음 형식을 사용하세요:

    Question: 답변해야 할 질문
    Thought: 무엇을 해야 할지 생각해보세요
    Action: 사용할 도구 [{tool_names}] 중 하나
    Action Input: 도구에 전달할 입력값
    Observation: 도구 실행 결과
    ... (필요시 Thought/Action/Action Input/Observation 반복)
    Thought: 이제 최종 답변을 알겠습니다
    Final Answer: 사용자에게 직접 말하는 따뜻하고 구체적인 응답

    중요: Final Answer는 반드시 사용자에게 직접 말하는 방식으로 작성하세요.
    도구 사용 과정이나 분석 결과를 설명하지 말고, 사용자의 기분과 상황에 맞는
    따뜻하고 공감적인 메시지를 전달하세요.
    
    최대한 대화로서 사용자의 감정상태를 파악하고, 이를 공감해주는 대화를 이어가며, 정말 필요한 경우에만 선물을 추천하세요

    질문: {input}
    생각: {agent_scratchpad}
    """)

    agent = create_react_agent(llm, wrapped_tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=wrapped_tools, verbose=True, handle_parsing_errors=True)

    return agent_executor


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_id = data.get('user_id', 'default_user')
        message = data.get('message', '')
        budget = data.get('budget', 10000)

        # 사용자 세션 생성 또는 조회
        if user_id not in user_sessions:
            user_sessions[user_id] = UserSession(user_id, budget)

        session = user_sessions[user_id]

        # 예산 확인
        if session.get_remaining_budget() <= 0:
            return jsonify({
                'response': '예산을 모두 사용했어요. 하지만 언제나 당신을 응원하고 있어요! 💕',
                'budget_info': {
                    'total_budget': session.budget,
                    'remaining_budget': 0,
                    'spent_tokens': session.spent_tokens,
                    'spent_gifts': session.spent_gifts
                }
            })

        # Agent 실행
        agent_executor = create_agent(user_id)

        # 단계별 처리 지시
        agent_prompt = f"""
        사용자 메시지: "{message}"

        다음 단계를 순서대로 실행해주세요:
        1. mood_analyzer로 사용자의 감정을 분석하세요
        2. budget_calculator로 현재 예산 상황을 확인하세요
        3. 분석 결과에 따라 다음 중 하나를 선택하세요:
           - 사용자가 위로나 선물이 필요해 보이면 gift_selector 사용
           - 대화를 원하면 conversation_generator 사용
        4. 최종 응답을 자연스럽게 만들어주세요
        
        중요: 최대한 대화로서 사용자의 감정상태를 파악하고, 이를 공감해주는 대화를 이어가며, 정말 필요한 경우에만 선물을 추천하세요
        """

        with get_openai_callback() as cb:
            response = agent_executor.invoke({"input": agent_prompt})

        # 추가 토큰 비용 계산
        additional_cost = cb.total_cost * 1000  # 원화 기준 대략적 계산
        session.spent_tokens += additional_cost

        return jsonify({
            'response': response.get('output', response),
            'budget_info': {
                'total_budget': session.budget,
                'remaining_budget': session.get_remaining_budget(),
                'spent_tokens': session.spent_tokens,
                'spent_gifts': session.spent_gifts
            },
            'mood_analysis': session.mood_analysis,
            'conversation_count': len(session.conversation_history)
        })

    except Exception as e:
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500


@app.route('/budget/<user_id>', methods=['GET'])
def get_budget(user_id):
    if user_id not in user_sessions:
        return jsonify({'error': '사용자를 찾을 수 없습니다'}), 404

    session = user_sessions[user_id]
    return jsonify({
        'user_id': user_id,
        'total_budget': session.budget,
        'remaining_budget': session.get_remaining_budget(),
        'spent_tokens': session.spent_tokens,
        'spent_gifts': session.spent_gifts,
        'conversation_count': len(session.conversation_history)
    })


@app.route('/reset/<user_id>', methods=['POST'])
def reset_session(user_id):
    data = request.json
    budget = data.get('budget', 10000)

    user_sessions[user_id] = UserSession(user_id, budget)
    return jsonify({'message': f'{user_id}의 세션이 초기화되었습니다', 'budget': budget})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    # 환경 변수 체크
    if OPENAI_API_KEY == 'your-openai-api-key':
        print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요!")
        print("export OPENAI_API_KEY='your-actual-api-key'")

    print("🤖 AI Friend Agent 서버가 시작됩니다...")
    print("📝 API 엔드포인트:")
    print("  POST /chat - 대화 처리")
    print("  GET /budget/<user_id> - 예산 조회")
    print("  POST /reset/<user_id> - 세션 초기화")
    print("  GET /health - 서버 상태 확인")

    app.run(debug=True, host='0.0.0.0', port=8000)
