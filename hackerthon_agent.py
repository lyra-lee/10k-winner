import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, ClassVar, Union
import re
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.manager import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from openai import OpenAI
from pydantic import BaseModel, Field

# --- 1. Flask 앱 및 환경 설정 ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 환경 변수 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key')
if OPENAI_API_KEY == 'your-openai-api-key':
    print("⚠️  OPENAI_API_KEY 환경 변수를 설정해주세요! `export OPENAI_API_KEY='your-actual-api-key'`")

# OpenAI 클라이언트 초기��
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
            profile.conversation_starters = analysis_data.get('conversation_starters', [])

            # 전역 프로필 및 사용자 세션에 저장
            ai_friend_profiles[profile.agent_id] = profile
            if user_id and user_id in user_sessions:
                user_sessions[user_id].created_agents[profile.agent_id] = profile

            print(f"""
🤖 AI 친구 '{profile.name}' 생성 완료!

👤 이름: {profile.name}
🎭 성격: {profile.personality}
💬 대화 스타일: {profile.conversation_style}
❤️ 관심사: {', '.join(profile.interests)}
🗣️ 대화 시작 문구들: {len(profile.conversation_starters)}개 준비됨

Agent ID: {profile.agent_id}
이제 상대방이 이 AI 친구와 대화할 수 있어요!
            """)

            result = {
                'agent_id': profile.agent_id,
                'name': profile.name,
                'personality': profile.personality,
                'conversation_style': profile.conversation_style,
                'interests': profile.interests,
                'conversation_starters': profile.conversation_starters
            }
            return json.dumps(result, ensure_ascii=False)
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

        # 예�� 내에서 선물 필터링
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
# --- Socket.IO 콜백 핸들러 ---
class SocketCallbackHandler(BaseCallbackHandler):
    """Agent의 실행 과정을 Socket.IO를 통해 클라이언트로 전송하는 콜백 핸들러"""

    def __init__(self, socketio_instance, sid: str):
        self.socketio = socketio_instance
        self.sid = sid

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """LLM이 실행을 시작할 때 호출됩니다."""
        # self.socketio.emit('create_ai_friend_action', {'message': '🤔 생각 중...'}, to=self.sid)

    def on_agent_action(
            self, action: AgentAction, color: Union[str, None] = None, **kwargs: Any
    ) -> Any:
        """Agent가 Action을 취할 때 호출됩니다. (수정된 부분)"""
        thought_log = action.log.strip()

        self.socketio.emit('create_ai_friend_action', {
            'thought': thought_log,  # 'thought' 키에 전체 로그를 전달
            # 'action': {'tool': action.tool, 'tool_input': action.tool_input}
        }, to=self.sid)

    def on_tool_end(
            self, output: str, color: Union[str, None], **kwargs: Any
    ) -> Any:
        """도구 실행이 끝났을 때 호출됩니다."""
        # self.socketio.emit('create_ai_friend_action', {'observation': output}, to=self.sid)

    def on_agent_finish(
            self, finish: AgentFinish, color: Union[str, None] = None, **kwargs: Any
    ) -> Any:
        """Agent가 최종 답변을 반환할 때 호출됩니다."""
        final_answer = finish.return_values['output']
        # self.socketio.emit('create_ai_friend_action', {'final_answer': final_answer}, to=self.sid)


def create_friend_creator_agent(user_id: str, socketio_instance, sid: str):
    """AI 친구 '생성'을 전담하는 에이전트"""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, api_key=OPENAI_API_KEY)
    tools = [
        RelationshipAnalyzerTool(),
        AIFriendCreatorTool(),
        ConversationHistoryTool()
    ]
    wrapped_tools = [UserIdToolWrapper(name=t.name, description=t.description, tool=t, user_id=user_id) for t in tools]

    from langchain.prompts import PromptTemplate
    prompt = PromptTemplate.from_template("""

        당신은 사용자를 위해 맞춤형 AI 친구를 만들어주는 전문가입니다.
        사용자가 제공한 정보를 바탕으로, 다음 단계를 엄격하게 순서대로 실행하여 AI 친구를 생성해야 합니다.

        사용 가능한 도구:
        {tools}

        다음 단계를 순서대로 실행하세요:
        1. relationship_analyzer로 상대방 정보 분석
        2. conversation_history_analyzer로 이전 대화 기록 분석 (있다면)
        3. ai_friend_creator로 맞춤형 AI 친구 생성


        **당신은 반드시 아래 설명된 생각/행동/관찰 사이클을 따라야 합니다.**

        응답 형식은 다음과 같습니다.

        Question: 사용자의 원래 질문
        Thought: 현재 상황을 분석하고, 다음 단계로 무엇을 해야 할지 결정합니다. 여기서 최종 답변을 생성해서는 안 됩니다.
        Action: 사용해야 할 도구의 이름. [{tool_names}] 중 하나여야 합니다.
        Action Input: 위 Action에서 선택한 도구에 전달할 입력값입니다.
        Observation: 이전 Action을 실행한 결과입니다. (이 부분은 시스템에 의해 채워집니다)

        ... (이 Thought/Action/Action Input/Observation 사이클은 필요한 만큼 반복될 수 있습니다) ...

        Thought: 이제 모든 정보가 수집되었고, 최종 답변을 할 준비가 되었습니다.
        Final Answer: 생성된 AI 친구에 대한 최종적이고 완전한 설명. 사용자가 받게 될 최종 응답입니다.

        **매우 중요한 규칙:**
        - 당신의 모든 응답은 'Thought:'로 시작해야 합니다.
        - 'Thought:' 다음에는 반드시 'Action:' 또는 'Final Answer:'가 와야 ���니다. 절대로 다른 텍스트를 생성해서는 안됩니다.
        - 도구를 사용할 필요가 없다고 판단되면, 즉시 'Final Answer:'를 제공하세요. 하지만 친구 생성 과정에서는 반드시 도구를 사용해야 합니다.

        중요: 상대방의 성격, 관심사, 관계 특성을 모두 고려하여 정말 그 사람에게 맞는 AI 친구를 만들어주세요.

        이제 시작하겠습니다.

        Question: {input}
        {agent_scratchpad}
        """)
    socket_callback = SocketCallbackHandler(socketio_instance, sid)
    agent = create_react_agent(llm, wrapped_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=wrapped_tools,
        verbose=True,
        handle_parsing_errors=True,
        callbacks=[socket_callback]
    )

    return agent_executor


def create_chat_agent(user_id: str, agent_id: str):
    """생성된 AI 친구와 '대화'를 전담하는 에이전트"""
    if agent_id not in ai_friend_profiles:
        raise ValueError("존재하지 않는 AI 친구(Agent) ID입니다.")

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.8, api_key=OPENAI_API_KEY)

    tools = [BudgetCalculatorTool(), MoodAnalyzerTool(), GiftSelectorTool(), ConversationTool(agent_id=agent_id)]
    wrapped_tools = [UserIdToolWrapper(name=t.name, description=t.description, tool=t, user_id=user_id) for t in tools]

    from langchain.prompts import PromptTemplate
    profile = ai_friend_profiles[agent_id]  # agent_id로 프로필을 가져옵니다.

    prompt = PromptTemplate.from_template(f"""
            당신은 이제부터 AI 친구 '{profile.name}'입니다. 당신의 주된 역할은 사용자와 자연스럽게 대화하는 ��입니다.
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


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


# --- WebSocket 이벤트 핸들러 ---
@socketio.on('connect')
def ws_connect():
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'WebSocket connection established'})


def run_agent_task(user_id, user_input, sid):
    """백그라운드에서 Agent를 실행하는 함수"""
    try:
        agent_executor = create_friend_creator_agent(user_id, socketio, sid)
        agent_executor.invoke({"input": user_input})
    except Exception as e:
        print(f"Agent 실행 중 오류 발생: {e}")
        emit('error', {'error': str(e)}, to=sid)


@socketio.on('create_ai_friend')
def ws_create_ai_friend(data):
    user_id = data.get('user_id')
    target_person_info = data.get('target_person_info')
    conversation_history = data.get('conversation_history', '')

    sid = request.sid

    if not user_id or not target_person_info:
        emit('create_ai_friend_response', {'error': 'user_id and target_person_info are required'})
        return
    if user_id not in user_sessions:
        budget = data.get('budget', 10000)
        user_sessions[user_id] = UserSession(user_id, budget)

    agent_executor = create_friend_creator_agent(user_id, socketio, sid)

    prompt = f"""
        사용자가 다음 상대방을 위한 AI 친구를 만들고 싶어합니다:

        상대방 정보: {target_person_info}
        이전 대화 기록: {conversation_history if conversation_history else '없음'}

        이 정보를 바탕으로 상대방에게 딱 맞는 AI 친구를 생성해주세요.

        최종 응답은 다음 형식으로 작성되어야 합니다:
        {{
            "agent_id": "생성된 AI 친구의 고유 ID"
        }}
        """

    with get_openai_callback() as cb:
        response = agent_executor.invoke({"input": prompt})

    output = response.get('output', '')

    try:
        data = json.loads(output)
        agent_id = data['agent_id']
        profile = ai_friend_profiles[agent_id]
        emit('create_ai_friend_response', {
            'agent_id': profile.agent_id,
            'name': profile.name,
            'personality': profile.personality,
        })
    except Exception:
        print("AI friend create JSON parse failed, output:", output)
        emit('create_ai_friend_response', {'response': output, 'cost': cost})


@socketio.on('chat')
def ws_chat(data):
    user_id = data.get('user_id')
    agent_id = data.get('agent_id')
    message = data.get('message')
    if not all([user_id, agent_id, message]):
        emit('chat_response', {'error': 'user_id, agent_id, message are required'})
        return
    session = user_sessions.get(user_id)
    if not session:
        emit('chat_response', {'error': 'User session not found'})
        return
    if not session.can_afford(50):
        emit('chat_response', {'response': '예산을 모두 사용했어요. 하지만 언제나 당신을 응원하고 있어요! 💕'})
        return
    agent_executor = create_chat_agent(user_id, agent_id)
    with get_openai_callback() as cb:
        response = agent_executor.invoke({"input": message})
        cost = cb.total_cost * 1300
        session.spent_tokens += cost
    # notify client of action completion
    emit('chat_action', {'status': 'completed', 'message': f"Action completed with cost {cost:.2f}"})
    ai_response = response.get('output', '미안해요, 지금은 답장할 수 없어요..')
    session.add_conversation(agent_id, message, ai_response, cost)
    emit('chat_response', {
        'response': ai_response,
        'budget_info': {
            'remaining_budget': session.get_remaining_budget(),
            'spent_total': session.spent_tokens + session.spent_gifts
        }
    })


if __name__ == '__main__':
    print("🤖 통합 AI 친구 서버가 WebSocket 모드로 시작됩니다...")
    socketio.run(app, debug=True, host='0.0.0.0', port=8000)
