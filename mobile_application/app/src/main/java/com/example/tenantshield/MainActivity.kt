package com.example.tenantshield

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.Gavel
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Login
import androidx.compose.material.icons.outlined.Logout
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.PersonSearch
import androidx.compose.material.icons.outlined.Videocam
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.activity.compose.BackHandler
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import kotlinx.coroutines.delay
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.firebase.auth.FirebaseAuth
import com.example.tenantshield.agents.orchestrator.OrchestratorState
import com.example.tenantshield.ui.screens.auth.LoginScreen
import com.example.tenantshield.ui.screens.home.HomeScreen
import com.example.tenantshield.ui.screens.inspect.InspectScreen
import com.example.tenantshield.ui.screens.owners.OwnersScreen
import com.example.tenantshield.ui.screens.profile.ProfileScreen
import com.example.tenantshield.ui.screens.reports.ReportsScreen
import com.example.tenantshield.ui.theme.OnSurfaceVariant
import com.example.tenantshield.ui.theme.Primary
import com.example.tenantshield.ui.theme.TenantShieldTheme
import com.example.tenantshield.ui.viewmodel.InspectionViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            TenantShieldTheme {
                TenantShieldApp()
            }
        }
    }
}

enum class AppDestination(
    val label: String,
    val icon: ImageVector,
) {
    HOME("Home", Icons.Outlined.Home),
    INSPECT("Inspect", Icons.Outlined.Videocam),
    OWNERS("Owners", Icons.Outlined.PersonSearch),
    REPORTS("Reports", Icons.Outlined.Description),
}

// Screens that are not in the bottom nav
enum class OverlayScreen {
    NONE, LOGIN, PROFILE
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TenantShieldApp(inspectionViewModel: InspectionViewModel = viewModel()) {
    val context = LocalContext.current
    LaunchedEffect(Unit) { inspectionViewModel.initialize(context) }

    var isLoggedIn by rememberSaveable {
        mutableStateOf(FirebaseAuth.getInstance().currentUser != null)
    }

    val inspectionUiState by inspectionViewModel.uiState.collectAsState()
    var currentDestination by rememberSaveable { mutableStateOf(AppDestination.HOME) }
    var previousDestination by rememberSaveable { mutableStateOf<AppDestination?>(null) }
    var overlayScreen by rememberSaveable { mutableStateOf(OverlayScreen.NONE) }
    var showLoginWarning by remember { mutableStateOf(false) }
    val showServerBusy = inspectionUiState.showServerBusy

    // Auto-dismiss login warning after 1 second, then redirect to login
    LaunchedEffect(showLoginWarning) {
        if (showLoginWarning) {
            delay(1000)
            showLoginWarning = false
            overlayScreen = OverlayScreen.LOGIN
        }
    }

    // Auto-navigate to Reports when complaint is generated
    LaunchedEffect(inspectionUiState.pipelineState) {
        if (inspectionUiState.pipelineState == OrchestratorState.COMPLETE
            && inspectionUiState.complaintForm != null) {
            previousDestination = currentDestination
            currentDestination = AppDestination.REPORTS
        }
    }

    // Back handler
    BackHandler(enabled = overlayScreen != OverlayScreen.NONE || currentDestination != AppDestination.HOME) {
        if (overlayScreen != OverlayScreen.NONE) {
            overlayScreen = OverlayScreen.NONE
        } else {
            val target = previousDestination ?: AppDestination.HOME
            previousDestination = null
            currentDestination = target
        }
    }

    fun navigateTo(dest: AppDestination) {
        if (dest != currentDestination) {
            previousDestination = currentDestination
            currentDestination = dest
        }
    }

    // Overlay screens (Login, Profile) — shown on top, back returns to previous
    if (overlayScreen == OverlayScreen.LOGIN) {
        LoginScreen(onLoginSuccess = {
            isLoggedIn = true
            overlayScreen = OverlayScreen.NONE
        })
        return
    }

    if (overlayScreen == OverlayScreen.PROFILE) {
        ProfileScreen(
            inspectionViewModel = inspectionViewModel,
            onBack = { overlayScreen = OverlayScreen.NONE },
            onSignOut = {
                FirebaseAuth.getInstance().signOut()
                isLoggedIn = false
                overlayScreen = OverlayScreen.NONE
            }
        )
        return
    }

    // Main app scaffold
    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            if (currentDestination != AppDestination.INSPECT) {
                TenantShieldTopBar(
                    isLoggedIn = isLoggedIn,
                    onProfileClick = {
                        overlayScreen = OverlayScreen.PROFILE
                    },
                    onLoginClick = {
                        overlayScreen = OverlayScreen.LOGIN
                    }
                )
            }
        },
        bottomBar = {
            if (currentDestination != AppDestination.INSPECT) {
                TenantShieldBottomBar(currentDestination) { dest ->
                    if (!isLoggedIn && (dest == AppDestination.INSPECT || dest == AppDestination.REPORTS)) {
                        showLoginWarning = true
                    } else {
                        navigateTo(dest)
                    }
                }
            }
        },
        floatingActionButton = {}
    ) { innerPadding ->
        when (currentDestination) {
            AppDestination.HOME -> HomeScreen(
                modifier = Modifier.padding(innerPadding),
                onStartInspection = {
                    if (isLoggedIn) {
                        inspectionViewModel.startInspectionFlow()
                        navigateTo(AppDestination.INSPECT)
                    } else {
                        showLoginWarning = true
                    }
                }
            )
            AppDestination.INSPECT -> InspectScreen(
                modifier = Modifier.fillMaxSize(),
                inspectionViewModel = inspectionViewModel,
                onClose = {
                    val target = previousDestination ?: AppDestination.HOME
                    previousDestination = null
                    currentDestination = target
                }
            )
            AppDestination.OWNERS -> OwnersScreen(Modifier.padding(innerPadding))
            AppDestination.REPORTS -> ReportsScreen(
                modifier = Modifier.padding(innerPadding)
            )
        }

        // Login warning popup
        AnimatedVisibility(
            visible = showLoginWarning,
            enter = fadeIn(),
            exit = fadeOut(),
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .padding(horizontal = 32.dp)
                        .fillMaxWidth()
                        .background(
                            Primary.copy(alpha = 0.95f),
                            RoundedCornerShape(12.dp)
                        )
                        .padding(24.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Please log in first",
                        style = MaterialTheme.typography.titleMedium,
                        color = Color.White,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
        // Server busy popup
        if (showServerBusy) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .background(Color.Black.copy(alpha = 0.5f)),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .padding(horizontal = 32.dp)
                        .fillMaxWidth()
                        .background(Color.White, RoundedCornerShape(16.dp))
                        .padding(32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Oops!",
                            style = MaterialTheme.typography.headlineMedium,
                            color = Primary,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = "There are so many users using the AI agent on the server right now. Please come back later again. Sorry!",
                            style = MaterialTheme.typography.bodyMedium,
                            color = OnSurfaceVariant,
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(24.dp))
                        Button(
                            onClick = {
                                inspectionViewModel.cancelSession()
                                currentDestination = AppDestination.HOME
                            },
                            shape = RoundedCornerShape(8.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Primary,
                                contentColor = Color.White
                            )
                        ) {
                            Text(
                                text = "GO BACK HOME",
                                style = MaterialTheme.typography.labelLarge,
                                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TenantShieldTopBar(
    isLoggedIn: Boolean = false,
    onProfileClick: () -> Unit = {},
    onLoginClick: () -> Unit = {}
) {
    TopAppBar(
        title = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Outlined.Gavel,
                    contentDescription = null,
                    tint = Primary,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(Modifier.width(12.dp))
                Text(
                    "TenantShield",
                    style = MaterialTheme.typography.headlineMedium.copy(fontSize = 22.sp),
                    color = Primary,
                    fontWeight = FontWeight.Bold
                )
            }
        },
        actions = {
            if (isLoggedIn) {
                IconButton(onClick = onProfileClick) {
                    Icon(
                        Icons.Outlined.Person,
                        contentDescription = "Profile",
                        tint = Primary
                    )
                }
            } else {
                IconButton(onClick = onLoginClick) {
                    Icon(
                        Icons.Outlined.Login,
                        contentDescription = "Login",
                        tint = Primary
                    )
                }
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    )
}

@Composable
private fun TenantShieldBottomBar(
    selected: AppDestination,
    onSelect: (AppDestination) -> Unit
) {
    NavigationBar(
        containerColor = Color.White.copy(alpha = 0.95f),
        tonalElevation = 0.dp,
    ) {
        AppDestination.entries.forEach { dest ->
            NavigationBarItem(
                icon = { Icon(dest.icon, dest.label) },
                label = {
                    Text(
                        dest.label.uppercase(),
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontSize = 11.sp,
                            letterSpacing = 1.sp
                        )
                    )
                },
                selected = dest == selected,
                onClick = { onSelect(dest) },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = Color.White,
                    selectedTextColor = Primary,
                    indicatorColor = Primary.copy(alpha = 0.9f),
                    unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                )
            )
        }
    }
}
