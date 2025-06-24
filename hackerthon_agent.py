import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, ClassVar

from flask import Flask, request, jsonify
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.manager import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

# --- 1. Flask 앱 및 환경 설정 ---
app = Flask(__name__)

# 환경 변수 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')
if OPENAI_API_KEY == 'your-openai-api-key':
    print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요! `export OPENAI_API_KEY='your-actual-api-key'`")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)


# --- 2. 통합 데이터 모델 및 전역 상태 관리 ---

class AIFriendProfile:
    """생성된 AI 친구의 프로필을 저장하는 클래스"""

    def __init__(self):
        self.agent_id = str(uuid.uuid4())
        self.name = "친구"
        self.personality = "다정함"
        self.conversation_style = "친근한 말투"
        self.interests = []
        self.special_memories = []
        self.relationship_context = "친구"
        self.gift_preferences = {}
        self.conversation_starters = []
        self.created_at = datetime.now().isoformat()


class UserSession:
    """사용자의 세션 정보를 관리하는 클래스 (생성 및 대화 기능 통합)"""

    def __init__(self, user_id: str, budget: float = 10000):
        self.user_id = user_id
        self.budget = budget
        self.spent_tokens = 0
        self.spent_gifts = 0
        self.conversation_history: Dict[str, List] = {}  # agent_id 별 대화 기록
        self.mood_analysis = {}
        self.created_agents: Dict[str, AIFriendProfile] = {}  # 사용자가 생성한 AI 친구 목록

    def get_remaining_budget(self) -> float:
        return self.budget - self.spent_tokens - self.spent_gifts

    def can_afford(self, cost: float) -> bool:
        return self.get_remaining_budget() >= cost

    def add_conversation(self, agent_id: str, user_msg: str, ai_msg: str, cost: float):
        if agent_id not in self.conversation_history:
            self.conversation_history[agent_id] = []
        self.conversation_history[agent_id].append({
            "user": user_msg,
            "ai": ai_msg,
            "timestamp": datetime.now().isoformat(),
            "cost": cost
        })


# 전역 저장소: 실제 프로덕션 환경에서는 DB 또는 Redis 사용을 권장합니다.
user_sessions: Dict[str, UserSession] = {}
ai_friend_profiles: Dict[str, AIFriendProfile] = {}


# --- 3. 공통 도구 및 래퍼 클래스 ---

class UserIdToolWrapper(BaseTool):
    """기존 도구에 user_id를 추가로 전달하기 위한 래퍼 클래스."""
    name: str
    description: str
    tool: BaseTool
    user_id: str

    def _run(self, query: str) -> str:
        return self.tool._run(query, user_id=self.user_id)

    async def _arun(self, query: str) -> str:
        if hasattr(self.tool, '_arun'):
            return await self.tool._arun(query, user_id=self.user_id)
        return self._run(query)


# --- 4. AI 친구 '생성' 관련 도구들 ---

class RelationshipAnalyzerTool(BaseTool):
    name: str = "relationship_analyzer"
    description: str = "사용자가 입력한 상대방 정보를 분석하여 관계성과 상대방의 성격을 파악하는 도구입니다."

    def _run(self, relationship_info: str, user_id: str = None) -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """당신은 관계 분석 전문가입니다. 사용자가 제공한 상대방 정보를 분석하여 다음 JSON 형태로 응답하세요:
                     {
                         "relationship_type": "관계 유형 (친구/연인/가족/동료 등)",
                         "personality_traits": ["성격특성1", "성격특성2"],
                         "interests": ["관심사1", "관심사2"],
                         "communication_style": "대화 스타일 설명",
                         "emotional_needs": ["감정적 니즈1", "감정적 니즈2"],
                         "gift_preferences": {"스티커": 5, "커피쿠폰": 8},
                         "conversation_starters": ["대화시작1", "대화시작2"]
                     }"""},
                    {"role": "user", "content": f"다음 정보를 분석해주세요: {relationship_info}"}
                ], max_tokens=800, temperature=0.3
            )
            return f"관계 분석 완료: {response.choices[0].message.content}"
        except Exception as e:
            return f"관계 분석 중 오류 발생: {str(e)}"


class ConversationHistoryTool(BaseTool):
    name: str = "conversation_history_analyzer"
    description: str = "이전 대화 내용을 분석하여 상대방에 대한 추가 정보를 파악하는 도구입니다."

    def _run(self, conversation_history: str, user_id: str = None) -> str:
        try:
            if not conversation_history.strip():
                return "대화 기록이 없어서 기본 설정으로 진행합니다."

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": """대화 기록을 분석하여 상대방의 추가적인 특성을 파악해주세요. 다음 JSON 형태로 응답하세요:
                     {
                         "recent_mood_patterns": ["최근 기분 패턴들"],
                         "conversation_preferences": "선호하는 대화 주제나 방식",
                         "special_moments": ["특별한 추억이나 언급사항들"],
                         "current_concerns": ["현재 관심사나 걱정거리들"],
                         "communication_frequency": "대화 빈도나 패턴"
                     }"""},
                    {"role": "user", "content": f"다음 대화 기록을 분석해주세요: {conversation_history}"}
                ],
                max_tokens=500,
                temperature=0.3
            )

            history_analysis = response.choices[0].message.content
            return f"대화 기록 분석 완료: {history_analysis}"

        except Exception as e:
            return f"대화 기록 분석 중 오류 발생: {str(e)}"

    def _arun(self, conversation_history: str, user_id: str = None):
        raise NotImplementedError("비동기 실행은 지원되지 않습니다.")


class AIFriendCreatorTool(BaseTool):
    name: str = "ai_friend_creator"
    description: str = "분석된 관계 정보를 바탕으로 맞춤형 AI 친구를 생성하는 도구입니다."

    def _run(self, analysis_result: str, user_id: str = None) -> str:
        try:
            json_part = analysis_result.split("관계 분석 완료: ", 1)[-1]
            analysis_data = json.loads(json_part.strip())

            profile = AIFriendProfile()

            name_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "상대방의 성격과 관계에 맞는 따뜻하고 친근한 한국어 AI 친구 이름을 2-3글자로 하나만 제안해주세요."},
                    {"role": "user",
                     "content": f"성격: {analysis_data.get('personality_traits', [])}, 관계: {analysis_data.get('relationship_type', '친구')}"}
                ], max_tokens=50, temperature=0.7
            )
            profile.name = name_response.choices[0].message.content.strip().replace('"', '')

            profile.personality = ", ".join(analysis_data.get('personality_traits', []))
            profile.conversation_style = analysis_data.get('communication_style', '일반적인 대화 스타일')
            profile.interests = analysis_data.get('interests', [])
            profile.relationship_context = analysis_data.get('relationship_type', '친구')
            profile.gift_preferences = analysis_data.get('gift_preferences', {})
            profile.conversation_starters = analysis_data.get('conversation_starters', [])

            # 전역 프로필 및 사용자 세션에 저장
            ai_friend_profiles[profile.agent_id] = profile
            if user_id and user_id in user_sessions:
                user_sessions[user_id].created_agents[profile.agent_id] = profile

            return f"""
🤖 AI 친구 '{profile.name}' 생성 완료!

👤 이름: {profile.name}
🎭 성격: {profile.personality}
💬 대화 스타일: {profile.conversation_style}
❤️ 관심사: {', '.join(profile.interests)}
🎁 선물 취향: 분석 완료
🗣️ 대화 시작 문구들: {len(profile.conversation_starters)}개 준비됨

Agent ID: {profile.agent_id}
이제 상대방이 이 AI 친구와 대화할 수 있어요!
            """
        except Exception as e:
            return f"AI 친구 생성 중 오류 발생: {str(e)}"


# --- 5. AI 친구와 '대화' 관련 도구들 ---

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
    description: str = "AI 친구의 페르소나에 맞춰 사용자와 자연스러운 대화를 생성합니다."
    agent_id: str

    def _run(self, user_message: str, user_id: str = None) -> str:
        # --- 세션 및 예산 확인 ---
        if not user_id or user_id not in user_sessions:
            return "오류: 사용자 세션을 찾을 수 없습니다."
        session = user_sessions[user_id]
        if not session.can_afford(50):  # 대화 최소 비용
            return "예산이 부족해서 더 이상 대화를 이어갈 수 없어요. 하지만 당신을 응원하고 있어요! 💕"

        # --- 페르소나 및 대화 기록 로드 ---
        if self.agent_id not in ai_friend_profiles:
            return "오류: 해당 AI 친구를 찾을 수 없습니다."
        profile = ai_friend_profiles[self.agent_id]

        # 이전 대화 기록을 가져와서 시스템 프롬프트에 포함
        history = session.conversation_history.get(self.agent_id, [])
        history_formatted = "\n".join(
            [f"사용자: {h['user']}\n{profile.name}: {h['ai']}" for h in history[-5:]])  # 최근 5개 대화

        # --- 페르소나를 적용한 프롬프트 생성 ---
        system_prompt = f"""당신은 이제부터 AI 친구 '{profile.name}'입니다. 당신의 정보는 다음과 같습니다:
    - 성격: {profile.personality}
    - 말투: {profile.conversation_style}
    - 관심사: {', '.join(profile.interests)}
    - 나와의 관계: '{profile.relationship_context}'

    이 페르소나에 완벽하게 몰입하여 사용자와 대화해주세요. 당신의 모든 답변은 '{profile.name}'으로서 하는 말이어야 합니다.
    아래는 사용자와의 최근 대화 기록입니다. 이 맥락을 이어서 자연스럽게 대화하세요.

    <최근 대화>
    {history_formatted}
    </최근 대화>
    """
        try:
            # --- OpenAI API 직접 호출 ---
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=1000,
            )
            ai_message = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            cost = tokens_used * 0.002
            session.spent_tokens += cost

            # UserSession에 정의된 add_conversation 메소드를 사용하도록 수정합니다.
            session.add_conversation(self.agent_id, user_message, ai_message, cost)

            return ai_message

        except Exception as e:
            print(f"대화 생성 중 오류 발생: {e}")
            return "미안해요, 지금은 답장을 보내기 조금 어려워요. 😥"


# --- 6. 에이전트 생성 함수들 ---

def create_friend_creator_agent(user_id: str):
    """AI 친구 '생성'을 전담하는 에이전트"""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, api_key=OPENAI_API_KEY)
    tools = [RelationshipAnalyzerTool(), AIFriendCreatorTool()]
    wrapped_tools = [UserIdToolWrapper(name=t.name, description=t.description, tool=t, user_id=user_id) for t in tools]

    prompt = hub.pull("hwchase17/react").partial(
        instructions="""당신은 사용자를 위해 맞춤형 AI 친구를 만들어주는 전문가입니다.
1. 'relationship_analyzer'로 상대방 정보를 분석하세요.
2. 분석된 결과를 바탕으로 'ai_friend_creator'를 호출하여 AI 친구를 생성하세요.
3. 최종적으로 생성된 친구의 이름과 Agent ID를 사용자에게 알려주세요."""
    )
    agent = create_react_agent(llm, wrapped_tools, prompt)
    return AgentExecutor(agent=agent, tools=wrapped_tools, verbose=True, handle_parsing_errors=True)


def create_chat_agent(user_id: str, agent_id: str):
    """생성된 AI 친구와 '대화'를 전담하는 에이전트"""
    if agent_id not in ai_friend_profiles:
        raise ValueError("존재하지 않는 AI 친구(Agent) ID입니다.")

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.8, api_key=OPENAI_API_KEY)

    tools = [BudgetCalculatorTool(), MoodAnalyzerTool(), GiftSelectorTool(), ConversationTool(agent_id=agent_id) ]
    wrapped_tools = [UserIdToolWrapper(name=t.name, description=t.description, tool=t, user_id=user_id) for t in tools]

    from langchain.prompts import PromptTemplate
    profile = ai_friend_profiles[agent_id]  # agent_id로 프로필을 가져옵니다.

    prompt = PromptTemplate.from_template(f"""
            당신은 이제부터 AI 친구 '{profile.name}'입니다. 당신의 주된 역할은 사용자와 자연스럽게 대화하는 것입니다.
            당신의 구체적인 성격({profile.personality})과 말투({profile.conversation_style})는 'conversation_generator' 도구에 완벽하게 정의되어 있습니다.

            사용자의 질문에 답변하기 위해 다음 도구들을 사용할 수 있습니다.
            - 일반적인 대화나 질문에는 'conversation_generator'를 사용하세요. 이것이 당신의 주된 소통 방식입니다.
            - 사용자에게 선물이 필요한 상태라면 'gift_selector' 도구를 사용하여 적절한 선물을 보내세요.

            {{tools}}

            다음 형식을 사용하세요:

            Question: 답변해야 할 질문
            Thought: 무엇을 해야 할지 생각합니다. 대부분의 경우 'conversation_generator' 도구를 사용해야 합니다.
            Action: 사용할 도구 [{{tool_names}}] 중 하나
            Action Input: 도구에 전달할 입력값
            Observation: 도구 실행 결과
            Thought: 이제 최종 답변을 알겠습니다.
            Final Answer: 'conversation_generator' 도구의 결과를 바탕으로, AI 친구 '{profile.name}'으로서 사용자에게 직접 말하는 것처럼 자연스럽고 따뜻하게 최종 답변을 전달합니다.

             **매우 중요한 규칙:**
            - 당신의 모든 응답은 'Thought:'로 시작해야 합니다.
            - 'Thought:' 다음에는 반드시 'Action:' 또는 'Final Answer:'가 와야 합니다. 절대로 다른 텍스트를 생성해서는 안됩니다.
            - 도구를 사용할 필요가 없다고 판단되면, 즉시 'Final Answer:'를 제공하세요. 하지만 친구 생성 과정에서는 반드시 도구를 사용해야 합니다.
    
            중요: 입력된 성격과 말투를 꼭 지켜주세요.
            
            질문: {{input}}
            생각: {{agent_scratchpad}}
        """)

    agent = create_react_agent(llm, wrapped_tools, prompt)

    return AgentExecutor(agent=agent, tools=wrapped_tools, verbose=True, handle_parsing_errors=True)


# --- 7. Flask API 엔드포인트 ---

@app.before_request
def ensure_user_session():
    """모든 요청 전에 사용자 세션을 확인하고 생성하는 함수"""
    if request.is_json:
        data = request.get_json()
        user_id = data.get('user_id')
        if user_id and user_id not in user_sessions:
            budget = data.get('budget', 10000)
            user_sessions[user_id] = UserSession(user_id, budget)


@app.route('/create-ai-friend', methods=['POST'])
def create_ai_friend():
    """AI 친구 생성을 요청하는 엔드포인트"""
    try:
        data = request.json
        user_id = data.get('user_id')
        if not user_id: return jsonify({'error': 'user_id가 필요합니다.'}), 400

        target_person_info = data.get('target_person_info', '')
        if not target_person_info: return jsonify({'error': 'target_person_info가 필요합니다.'}), 400

        agent_executor = create_friend_creator_agent(user_id)

        prompt = f"다음 정보를 가진 사람을 위한 AI 친구를 만들어주세요: {target_person_info}"

        with get_openai_callback() as cb:
            response = agent_executor.invoke({"input": prompt})
            cost = cb.total_cost * 1300  # 원화 기준 대략적 환산
            user_sessions[user_id].spent_tokens += cost

        # 응답에서 Agent ID 추출
        output = response.get('output', '')
        try:
            agent_id = output.split('Agent ID: ')[1].split(')')[0]
            created_profile = ai_friend_profiles[agent_id]
            final_message = f"🎉 AI 친구 '{created_profile.name}'님이 생성되었습니다! 이제 이 친구와 대화할 수 있어요."
            return jsonify({
                'message': final_message,
                'agent_id': created_profile.agent_id,
                'name': created_profile.name,
                'personality': created_profile.personality,
                'cost': cost
            })
        except (IndexError, KeyError) as e:
            return jsonify({'message': 'AI 친구 생성 완료', 'response': output, 'cost': cost})

    except Exception as e:
        return jsonify({'error': f'AI 친구 생성 중 오류: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """생성된 AI 친구와 대화하는 엔드포인트"""
    try:
        data = request.json
        user_id = data.get('user_id')
        agent_id = data.get('agent_id')
        message = data.get('message', '')

        if not all([user_id, agent_id, message]):
            return jsonify({'error': 'user_id, agent_id, message는 필수 항목입니다.'}), 400

        session = user_sessions.get(user_id)
        if not session: return jsonify({'error': '존재하지 않는 사용자입니다.'}), 404
        if not session.can_afford(50):  # 최소 대화 비용
            return jsonify({'response': '예산을 모두 사용했어요. 하지만 언제나 당신을 응원하고 있어요! 💕'})

        agent_executor = create_chat_agent(user_id, agent_id)

        with get_openai_callback() as cb:
            response = agent_executor.invoke({"input": message})
            cost = cb.total_cost * 1300  # 원화 기준 대략적 환산
            session.spent_tokens += cost

        ai_response = response.get('output', '미안해요, 지금은 답장할 수 없어요.')
        session.add_conversation(agent_id, message, ai_response, cost)

        return jsonify({
            'response': ai_response,
            'budget_info': {
                'remaining_budget': session.get_remaining_budget(),
                'spent_total': session.spent_tokens + session.spent_gifts
            }
        })

    except ValueError as e:  # 존재하지 않는 agent_id 처리
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'대화 처리 중 오류: {str(e)}'}), 500


# --- 8. 기타 유틸리티 엔드포인트 ---

@app.route('/user/<user_id>', methods=['GET'])
def get_user_info(user_id):
    """사용자 정보(예산, 생성한 친구 목록)를 반환"""
    if user_id not in user_sessions:
        return jsonify({'error': '사용자를 찾을 수 없습니다'}), 404

    session = user_sessions[user_id]
    created_agents_info = {
        agent_id: {"name": profile.name, "personality": profile.personality}
        for agent_id, profile in session.created_agents.items()
    }

    return jsonify({
        'user_id': user_id,
        'budget_info': {
            'total_budget': session.budget,
            'remaining_budget': session.get_remaining_budget(),
            'spent_tokens': session.spent_tokens,
            'spent_gifts': session.spent_gifts,
        },
        'created_ai_friends': created_agents_info
    })


@app.route('/reset/<user_id>', methods=['POST'])
def reset_session(user_id):
    budget = request.json.get('budget', 10000)
    user_sessions[user_id] = UserSession(user_id, budget)
    # 전역 프로필은 유지하되, 사용자의 생성 목록만 초기화
    return jsonify({'message': f'{user_id}의 세션이 초기화되었습니다', 'budget': budget})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    print("🤖 통합 AI 친구 서버가 시작됩니다...")
    print("📝 API 엔드포인트:")
    print("  POST /create-ai-friend - 새로운 AI 친구 생성")
    print("  POST /chat - 생성된 AI 친구와 대화")
    print("  GET /user/<user_id> - 사용자 정보 및 예산 조회")
    print("  POST /reset/<user_id> - 사용자 세션 초기화")
    print("  GET /health - 서버 상태 확인")
    app.run(debug=True, host='0.0.0.0', port=8000)