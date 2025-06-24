import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.manager import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, ClassVar
import uuid

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


ai_friend_profiles: Dict[str, AIFriendProfile] = {}
user_sessions: Dict[str, UserSession] = {}


class RelationshipAnalyzerTool(BaseTool):
    name: str = "relationship_analyzer"
    description: str = "사용자가 입력한 상대방 정보를 분석하여 관계성과 상대방의 성격을 파악하는 도구입니다."

    def _run(self, relationship_info: str, user_id: str = None) -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": """당신은 관계 분석 전문가입니다. 사용자가 제공한 상대방 정보를 분석하여 다음 JSON 형태로 응답하세요:
                     {
                         "relationship_type": "관계 유형 (친구/연인/가족/동료 등)",
                         "personality_traits": ["성격특성1", "성격특성2", "성격특성3"],
                         "interests": ["관심사1", "관심사2", "관심사3"],
                         "communication_style": "대화 스타일 설명",
                         "emotional_needs": ["감정적 니즈1", "감정적 니즈2"],
                         "gift_preferences": {
                             "스티커": 적합도점수(1-10),
                             "이모티콘": 적합도점수(1-10),
                             "커피쿠폰": 적합도점수(1-10),
                             "꽃다발": 적합도점수(1-10),
                             "초콜릿": 적합도점수(1-10),
                             "음악선물": 적합도점수(1-10)
                         },
                         "conversation_starters": ["대화시작1", "대화시작2", "대화시작3"]
                     }"""},
                    {"role": "user", "content": f"다음 정보를 분석해주세요: {relationship_info}"}
                ],
                max_tokens=800,
                temperature=0.3
            )

            analysis = response.choices[0].message.content
            return f"관계 분석 완료: {analysis}"

        except Exception as e:
            return f"관계 분석 중 오류 발생: {str(e)}"

    def _arun(self, relationship_info: str, user_id: str = None):
        raise NotImplementedError("비동기 실행은 지원되지 않습니다.")


class AIFriendCreatorTool(BaseTool):
    name: str = "ai_friend_creator"
    description: str = "분석된 관계 정보를 바탕으로 맞춤형 AI 친구를 생성하는 도구입니다."

    def _run(self, analysis_result: str, user_id: str = None) -> str:
        try:
            # --- 수정된 부분 시작 ---
            input_str = analysis_result

            # 입력값에 "관계 분석 완료: " 접두사가 있는지 확인
            if "관계 분석 완료: " in input_str:
                # 접두사가 있으면 분리해서 JSON 부분만 추출
                json_part = input_str.split("관계 분석 완료: ", 1)[1]
            else:
                # 접두사가 없으면 입력값 전체를 JSON으로 간주
                json_part = input_str

            # 공백이나 불필요한 문자를 제거하고 JSON 파싱
            analysis_data = json.loads(json_part.strip())
            # --- 수정된 부분 끝 ---

            # AI 친구 프로필 생성
            profile = AIFriendProfile()

            # 이름 생성 (오류 방지를 위해 choices 리스트 확인)
            name_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": "상대방의 성격과 관계에 맞는 따뜻하고 친근한 AI 친구 이름을 하나만 제안해주세요. 한국어 이름으로 2-3글자가 좋습니다."},
                    {"role": "user",
                     "content": f"성격: {analysis_data['personality_traits']}, 관계: {analysis_data.get('relationship_type', '친구')}"}
                ],
                max_tokens=50,
                temperature=0.7
            )

            # --- 이름 생성 부분 안정성 강화 ---
            if not name_response.choices:
                return "AI 친구 이름 생성에 실패했습니다. API 응답이 비어있습니다."

            profile.name = name_response.choices[0].message.content.strip()

            # 프로필 정보 설정 (get 메서드를 사용하여 특정 키가 없어도 오류 방지)
            profile.personality = ", ".join(analysis_data.get('personality_traits', []))
            profile.conversation_style = analysis_data.get('communication_style', '일반적인 대화 스타일')
            profile.interests = analysis_data.get('interests', [])
            profile.relationship_context = analysis_data.get('relationship_type', '친구')
            profile.gift_preferences = analysis_data.get('gift_preferences', {})
            profile.conversation_starters = analysis_data.get('conversation_starters', [])

            # 저장
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

        except json.JSONDecodeError as e:
            return f"AI 친구 생성 중 JSON 파싱 오류 발생: {str(e)}\n입력된 내용: {analysis_result}"
        except Exception as e:
            import traceback
            return f"AI 친구 생성 중 오류 발생: {str(e)}\n{traceback.format_exc()}"

    def _arun(self, analysis_result: str, user_id: str = None):
        raise NotImplementedError("비동기 실행은 지원되지 않습니다.")


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


def create_agent_creator(user_id: str):
    """AI 친구 생성을 위한 에이전트"""
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )

    tools = [
        RelationshipAnalyzerTool(),
        AIFriendCreatorTool(),
        ConversationHistoryTool()
    ]

    wrapped_tools = [
        UserIdToolWrapper(
            name=tool.name,
            description=tool.description,
            tool=tool,
            user_id=user_id
        ) for tool in tools
    ]

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
    - 'Thought:' 다음에는 반드시 'Action:' 또는 'Final Answer:'가 와야 합니다. 절대로 다른 텍스트를 생성해서는 안됩니다.
    - 도구를 사용할 필요가 없다고 판단되면, 즉시 'Final Answer:'를 제공하세요. 하지만 친구 생성 과정에서는 반드시 도구를 사용해야 합니다.
    
    중요: 상대방의 성격, 관심사, 관계 특성을 모두 고려하여 정말 그 사람에게 맞는 AI 친구를 만들어주세요.
    
    이제 시작하겠습니다.
    
    Question: {input}
    {agent_scratchpad}
    """)

    agent = create_react_agent(llm, wrapped_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=wrapped_tools, verbose=True, handle_parsing_errors=True)

    return agent_executor

@app.route('/create-ai-friend', methods=['POST'])
def create_ai_friend():
    """AI 친구 생성 요청"""
    try:
        data = request.json
        user_id = data.get('user_id', f'creator_{datetime.now().timestamp()}')
        target_person_info = data.get('target_person_info', '')
        conversation_history = data.get('conversation_history', '')

        agent_executor = create_agent_creator(user_id)

        prompt = f"""
        사용자가 다음 상대방을 위한 AI 친구를 만들고 싶어합니다:

        상대방 정보: {target_person_info}
        이전 대화 기록: {conversation_history if conversation_history else '없음'}

        이 정보를 바탕으로 상대방에게 딱 맞는 AI 친구를 생성해주세요.
        """

        with get_openai_callback() as cb:
            response = agent_executor.invoke({"input": prompt})

        return jsonify({
            'message': '🎉 AI 친구 생성 완료!',
            'response': response.get('output', response),
            'created_agents': list(session.created_agents.keys()),
        })

    except Exception as e:
        return jsonify({'error': f'AI 친구 생성 중 오류가 발생했습니다: {str(e)}'}), 500



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
