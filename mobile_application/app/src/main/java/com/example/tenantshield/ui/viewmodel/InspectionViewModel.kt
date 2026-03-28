package com.example.tenantshield.ui.viewmodel

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import com.example.tenantshield.agents.orchestrator.AgentOrchestrator
import com.example.tenantshield.agents.orchestrator.OrchestratorState
import com.example.tenantshield.agents.service.GeminiApiService
import com.example.tenantshield.agents.service.VoiceService
import com.example.tenantshield.agents.interacting.InteractingAgent
import com.example.tenantshield.agents.inspection.InspectionAgent
import com.example.tenantshield.agents.filing.FilingAgent
import com.example.tenantshield.agents.models.UserInfo
import com.example.tenantshield.agents.models.InspectionResult
import com.example.tenantshield.agents.models.ComplaintForm
import com.example.tenantshield.agents.firebase.FirebaseAuthService
import com.example.tenantshield.agents.firebase.FirestoreService
import com.example.tenantshield.agents.firebase.StorageService

data class ConversationMessage(
    val sender: String, // "agent" or "user"
    val text: String
)

class InspectionViewModel(application: Application) : AndroidViewModel(application) {

    data class UiState(
        val pipelineState: OrchestratorState = OrchestratorState.IDLE,
        val userInfo: UserInfo? = null,
        val inspectionResult: InspectionResult? = null,
        val complaintForm: ComplaintForm? = null,
        val currentVoiceText: String = "",
        val isListening: Boolean = false,
        val isSpeaking: Boolean = false,
        val errorMessage: String? = null,
        val showServerBusy: Boolean = false,
        val conversationMessages: List<ConversationMessage> = emptyList(),
        val lastUserTranscript: String = "", // Last STT result, editable by user
        val awaitingUserConfirmation: Boolean = false // True when user can edit transcript
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var geminiApiService: GeminiApiService? = null
    private var voiceService: VoiceService? = null
    private var orchestrator: AgentOrchestrator? = null
    private var initialized = false

    // Firebase services
    val authService = FirebaseAuthService()
    private val firestoreService = FirestoreService()
    private val storageService = StorageService()
    private var submittedImagePaths: List<String> = emptyList()

    fun initialize(context: Context) {
        if (initialized) return

        val gemini = GeminiApiService()
        val voice = VoiceService(context)
        geminiApiService = gemini
        voiceService = voice

        val interactingAgent = InteractingAgent(gemini, voice, context)
        val inspectionAgent = InspectionAgent(gemini)
        val filingAgent = FilingAgent(gemini)

        val orch = AgentOrchestrator(interactingAgent, inspectionAgent, filingAgent)
        orchestrator = orch

        orch.setListener(object : AgentOrchestrator.OrchestratorListener {
            override fun onStateChanged(newState: OrchestratorState) {
                _uiState.update { it.copy(pipelineState = newState, errorMessage = null) }
            }

            override fun onUserInfoCollected(info: UserInfo) {
                _uiState.update { it.copy(userInfo = info) }
            }

            override fun onInspectionComplete(result: InspectionResult) {
                _uiState.update { it.copy(inspectionResult = result) }
            }

            override fun onComplaintGenerated(form: ComplaintForm) {
                _uiState.update { it.copy(complaintForm = form) }
                // Save to Firebase when pipeline completes
                saveInspectionToFirebase()
            }

            override fun onError(agentId: String, message: String) {
                val isOverloaded = message.contains("503") || message.contains("overloaded")
                        || message.contains("high demand") || message.contains("RESOURCE_EXHAUSTED")
                        || message.contains("unavailable", ignoreCase = true)
                if (isOverloaded) {
                    _uiState.update { it.copy(showServerBusy = true, errorMessage = null) }
                } else {
                    _uiState.update { it.copy(errorMessage = "[$agentId] $message") }
                }
            }

            override fun onVoiceOutput(text: String) {
                _uiState.update { state ->
                    state.copy(
                        currentVoiceText = text,
                        isSpeaking = true,
                        conversationMessages = state.conversationMessages + ConversationMessage("agent", text)
                    )
                }
                voice.speak(text, object : VoiceService.VoiceListener {
                    override fun onSpeechResult(text: String) {}
                    override fun onSpeechError(errorCode: Int) {}
                    override fun onTtsComplete() {
                        _uiState.update { it.copy(isSpeaking = false) }
                        if (_uiState.value.pipelineState == OrchestratorState.COLLECTING_INFO) {
                            _uiState.update { it.copy(isListening = true) }
                            voice.startListening(object : VoiceService.VoiceListener {
                                override fun onSpeechResult(text: String) {
                                    _uiState.update { it.copy(
                                        isListening = false,
                                        lastUserTranscript = text,
                                        awaitingUserConfirmation = true
                                    ) }
                                }
                                override fun onSpeechError(errorCode: Int) {
                                    _uiState.update { it.copy(
                                        isListening = false,
                                        awaitingUserConfirmation = true,
                                        lastUserTranscript = ""
                                    ) }
                                }
                                override fun onTtsComplete() {}
                            })
                        }
                    }
                })
            }
        })

        initialized = true
    }

    /** User confirms the transcript (possibly edited) and sends it to the agent */
    fun confirmUserInput(text: String) {
        _uiState.update { state ->
            state.copy(
                awaitingUserConfirmation = false,
                lastUserTranscript = "",
                conversationMessages = state.conversationMessages + ConversationMessage("user", text)
            )
        }
        orchestrator?.processUserSpeech(text)
    }

    /** User wants to re-speak instead of using the transcript */
    fun retryListening() {
        _uiState.update { it.copy(awaitingUserConfirmation = false, lastUserTranscript = "") }
        voiceService?.let { voice ->
            _uiState.update { it.copy(isListening = true) }
            voice.startListening(object : VoiceService.VoiceListener {
                override fun onSpeechResult(text: String) {
                    _uiState.update { it.copy(
                        isListening = false,
                        lastUserTranscript = text,
                        awaitingUserConfirmation = true
                    ) }
                }
                override fun onSpeechError(errorCode: Int) {
                    _uiState.update { it.copy(
                        isListening = false,
                        awaitingUserConfirmation = true,
                        lastUserTranscript = ""
                    ) }
                }
                override fun onTtsComplete() {}
            })
        }
    }

    fun startInspectionFlow() {
        _uiState.update { UiState() }
        orchestrator?.startSession()
    }

    fun submitImages(paths: List<String>) {
        submittedImagePaths = paths
        orchestrator?.submitImages(paths)
    }

    fun retry() {
        _uiState.update { it.copy(errorMessage = null) }
        orchestrator?.retry()
    }

    fun cancelSession() {
        _uiState.update { UiState() }
        orchestrator?.cancelSession()
    }

    /** Saves the completed inspection to Firebase (user profile, images, inspection record) */
    private fun saveInspectionToFirebase() {
        val userId = authService.getUid() ?: return
        val state = _uiState.value
        val userInfo = state.userInfo ?: return
        val result = state.inspectionResult
        val form = state.complaintForm

        // 1. Save/update user profile
        firestoreService.saveUserProfile(userId, userInfo, object : FirestoreService.FirestoreCallback {
            override fun onSuccess() {
                android.util.Log.d("InspectionVM", "User profile saved")
            }
            override fun onError(message: String) {
                android.util.Log.e("InspectionVM", "Failed to save profile: $message")
            }
        })

        // 2. Upload images then save inspection record
        val inspectionId = "insp_${System.currentTimeMillis()}"
        if (submittedImagePaths.isNotEmpty()) {
            storageService.uploadMultipleImages(userId, inspectionId, submittedImagePaths,
                object : StorageService.UploadCallback {
                    override fun onSuccess(downloadUrls: List<String>) {
                        android.util.Log.d("InspectionVM", "Images uploaded: ${downloadUrls.size}")
                        saveInspectionRecord(userId, result, form, downloadUrls)
                    }
                    override fun onError(message: String) {
                        android.util.Log.e("InspectionVM", "Image upload failed: $message")
                        // Save record without image URLs
                        saveInspectionRecord(userId, result, form, emptyList())
                    }
                })
        } else {
            saveInspectionRecord(userId, result, form, emptyList())
        }
    }

    private fun saveInspectionRecord(
        userId: String,
        result: InspectionResult?,
        form: ComplaintForm?,
        imageUrls: List<String>
    ) {
        if (result == null) return
        firestoreService.saveInspectionRecord(userId, result, form, imageUrls,
            object : FirestoreService.FirestoreCallback {
                override fun onSuccess() {
                    android.util.Log.d("InspectionVM", "Inspection record saved to Firestore")
                }
                override fun onError(message: String) {
                    android.util.Log.e("InspectionVM", "Failed to save inspection: $message")
                }
            })
    }

    override fun onCleared() {
        super.onCleared()
        voiceService?.destroy()
    }
}
