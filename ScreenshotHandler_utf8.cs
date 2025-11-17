using UnityEngine;
using System.Collections;
using System.Net.Sockets;
using System;
using System.Collections.Generic;

public class ScreenshotHandler : MonoBehaviour
{
    // Socket Communication
    public Camera screenshotCamera;
    public RenderTexture screenshotRenderTexture;
    private Socket clientSocket;
    private const string SERVER_IP = "127.0.0.1";
    private const int SERVER_PORT = 8188;
    private bool isConnected = false;

    // Movement Parameters
    public float moveSpeed = 5.0f;
    public float turnSpeed = 90.0f; // degrees per second

    // Tunable Turn Angles for Discrete Actions (in degrees)
    public float smallTurnAngle = 15f;
    public float largeTurnAngle = 45f;

    // Reward Parameters (set these in Unity Inspector)
    [Header("Reward Settings")]
    public float progressScale = 2.0f;           // Distance-based shaping multiplier
    public float stepPenalty = -0.01f;           // Negative reward each step
    public float headingPenaltyScale = 0.01f;    // Penalize large misalignment to target
    public float targetBonus = 1000.0f;          // Bonus for actually reaching the target
    public float collisionPenalty = -1000.0f;    // Penalty for collision

    [Header("Partial Success Settings")]
    public bool usePartialSuccessThreshold = true;
    public float partialSuccessThreshold = 1.5f; // Distance within which we consider partial success
    public float partialSuccessFraction = 0.8f;  // Fraction of targetBonus given for partial success

    [Header("Episode Settings")]
    public int maxSteps = 100;              // Max steps before forced episode end
    public float episodeTimeLimit = 40.0f;  // Seconds before forced episode end
    public float frameInterval = 0.2f;      // Seconds between each RL step

    // Additional Reward Parameters for Enhanced Approach
    // (These parameters are used in the reward shaping logic.)
    public float timePenalty = -0.005f;             // Extra time penalty per step.
    public float smoothnessPenaltyScale = 0.02f;      // Penalty for large steering changes.
    private float lastPotential = 0.0f;               // For potential-based shaping.
    private float lastSteeringAngle = 0.0f;           // To track smoothness.

    // Internal State
    private float cumulativeReward = 0.0f;
    private bool collisionOccurred = false;
    private bool targetReached = false;
    private bool timeLimitReached = false;
    private float timeSinceLastFrame = 0f;
    private int stepCounter = 0;
    private float episodeStartTime;
    private bool isResettingEpisode = false;

    // Distance Tracking
    private float lastDistance;

    // Environment References
    public Transform target;
    private Vector3 initialPosition;
    public Transform agentTransform;
    public Vector2 terminationBoxSize = new Vector2(2.0f, 2.0f);

    // Path Visualization
    public LineRenderer pathLineRenderer;
    private List<Vector3> pathPositions = new List<Vector3>();

    // Boundary Definitions (corners of your environment)
    private Vector3 p1 = new Vector3(20.4f, 0f, 22.6f);
    private Vector3 p2 = new Vector3(83.2f, 0f, 22.6f);
    private Vector3 p3 = new Vector3(83.2f, 0f, -59.6f);
    private Vector3 p4 = new Vector3(20.4f, 0f, -59.6f);

    // Collision Flag
    private float collision_flag = 0.0f;

    // LineRenderer Colors
    private List<Color> lineColors = new List<Color>()
    {
        Color.red,
        Color.green,
        Color.blue,
        Color.yellow,
        Color.cyan,
        Color.magenta,
        Color.white,
        Color.black
    };
    private int currentColorIndex = 0;

    // Store last image data (if needed)
    private byte[] lastImageData = null;

    void Start()
    {
        // Disable interfering FPS controller if present
        var fpsController = agentTransform.GetComponent<UnityStandardAssets.Characters.FirstPerson.RigidbodyFirstPersonController>();
        if (fpsController != null)
        {
            fpsController.enabled = false;
            Debug.Log("Disabled FPS controller for algorithm control.");
        }

        ConnectToServer();

        // Record initial agent position and initialize timer
        initialPosition = agentTransform.position;
        episodeStartTime = Time.time;

        // Initialize line renderer if assigned
        if (pathLineRenderer != null)
        {
            pathLineRenderer.positionCount = 0;
            pathLineRenderer.widthMultiplier = 0.2f;
            pathLineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            pathLineRenderer.startColor = lineColors[currentColorIndex];
            pathLineRenderer.endColor = lineColors[currentColorIndex];
            currentColorIndex = (currentColorIndex + 1) % lineColors.Count;
            pathLineRenderer.gameObject.layer = LayerMask.NameToLayer("PathLine");
            int pathLineLayerMask = 1 << LayerMask.NameToLayer("PathLine");
            screenshotCamera.cullingMask &= ~pathLineLayerMask;
        }
        else
        {
            Debug.LogWarning("No initial LineRenderer assigned. A new one will be created on episode reset.");
        }

        AddPositionToPath(agentTransform.position);

        // Initialize lastDistance and shaping parameters
        lastDistance = Vector3.Distance(agentTransform.position, target.position);
        lastPotential = -lastDistance;
        lastSteeringAngle = agentTransform.eulerAngles.y;
    }

    void ConnectToServer()
    {
        try
        {
            clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            clientSocket.Connect(SERVER_IP, SERVER_PORT);
            isConnected = true;
            Debug.Log("Connected to server.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to connect: {e.Message}");
            isConnected = false;
        }
    }

    void Update()
    {
        if (!isConnected) return;

        // Check time limit
        if (!isResettingEpisode && !targetReached && (Time.time - episodeStartTime >= episodeTimeLimit))
        {
            Debug.Log("Time limit reached. Marking timeLimitReached and resetting agent (no target bonus).");
            timeLimitReached = true;
            ResetAgent(timeLimitReset: true);
        }

        // Execute RL step at fixed intervals
        timeSinceLastFrame += Time.deltaTime;
        if (timeSinceLastFrame >= frameInterval)
        {
            timeSinceLastFrame = 0f;
            StartCoroutine(CaptureSendAndReceive());
            stepCounter++;
        }

        // Check if agent left the boundaries
        if (!IsWithinBounds(agentTransform.position))
        {
            Debug.Log("Agent left boundaries. Ending episode.");
            cumulativeReward += collisionPenalty;
            collision_flag = 1.0f;
        }
    }

    void FixedUpdate()
    {
        // Check if agent has reached the target zone
        if (Mathf.Abs(agentTransform.position.x - target.position.x) < terminationBoxSize.x / 2 &&
            Mathf.Abs(agentTransform.position.z - target.position.z) < terminationBoxSize.y / 2)
        {
            targetReached = true;
            ResetAgent(timeLimitReset: false);
        }
    }

    private IEnumerator CaptureSendAndReceive()
    {
        yield return new WaitForEndOfFrame();

        screenshotCamera.targetTexture = screenshotRenderTexture;
        RenderTexture.active = screenshotRenderTexture;
        screenshotCamera.Render();

        Texture2D screenshot = new Texture2D(screenshotRenderTexture.width, screenshotRenderTexture.height, TextureFormat.RGB24, false);
        screenshot.ReadPixels(new Rect(0, 0, screenshotRenderTexture.width, screenshotRenderTexture.height), 0, 0);
        screenshot.Apply();

        screenshotCamera.targetTexture = null;
        RenderTexture.active = null;

        byte[] screenshotBytes = screenshot.EncodeToPNG();
        Destroy(screenshot);

        if (screenshotBytes != null)
        {
            lastImageData = screenshotBytes;
            SendData(screenshotBytes);
        }
        else
        {
            Debug.LogError("Failed to encode screenshot.");
        }
    }

    void SendData(byte[] data)
    {
        try
        {
            // Send image length prefix
            byte[] lengthPrefix = BitConverter.GetBytes(data.Length);
            if (BitConverter.IsLittleEndian) Array.Reverse(lengthPrefix);
            clientSocket.Send(lengthPrefix);

            // Send image data
            clientSocket.Send(data);

            // Send numeric state info
            float currentSpeed = GetCurrentSpeed();
            byte[] speedData = BitConverter.GetBytes(currentSpeed);
            if (BitConverter.IsLittleEndian) Array.Reverse(speedData);
            clientSocket.Send(speedData);

            Vector3 relativePosition = target.position - agentTransform.position;
            float deltaX = relativePosition.x;
            float deltaZ = relativePosition.z;
            byte[] deltaXData = BitConverter.GetBytes(deltaX);
            byte[] deltaZData = BitConverter.GetBytes(deltaZ);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(deltaXData);
                Array.Reverse(deltaZData);
            }
            clientSocket.Send(deltaXData);
            clientSocket.Send(deltaZData);

            Vector3 targetDirection = (target.position - agentTransform.position).normalized;
            float angle = Vector3.Angle(agentTransform.forward, targetDirection);
            byte[] angleData = BitConverter.GetBytes(angle);
            if (BitConverter.IsLittleEndian) Array.Reverse(angleData);
            clientSocket.Send(angleData);

            // Receive action code
            byte[] actionBuffer = new byte[4];
            int totalBytesReceived = 0;
            int bytesLeft = actionBuffer.Length;
            while (bytesLeft > 0)
            {
                int bytesRead = clientSocket.Receive(actionBuffer, totalBytesReceived, bytesLeft, SocketFlags.None);
                if (bytesRead == 0)
                {
                    Debug.LogError("Connection closed by server.");
                    isConnected = false;
                    return;
                }
                totalBytesReceived += bytesRead;
                bytesLeft -= bytesRead;
            }
            if (BitConverter.IsLittleEndian) Array.Reverse(actionBuffer);
            int actionCode = BitConverter.ToInt32(actionBuffer, 0);

            // Apply the received action
            ApplyAction(actionCode);

            // Compute reward and check terminal condition
            float doneFlag = 0.0f;
            float reward = GetReward();

            if (IsTerminalState())
            {
                doneFlag = 1.0f;
            }

            SendReward(reward, doneFlag);
            cumulativeReward += reward;
        }
        catch (Exception e)
        {
            Debug.LogError($"Send/Receive error: {e.Message}");
            isConnected = false;
        }
    }

    void SendReward(float reward, float doneFlag)
    {
        try
        {
            byte[] rewardData = BitConverter.GetBytes(reward);
            byte[] doneData = BitConverter.GetBytes(doneFlag);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(rewardData);
                Array.Reverse(doneData);
            }
            clientSocket.Send(rewardData);
            clientSocket.Send(doneData);

            if (doneFlag == 1.0f)
            {
                Debug.Log($"Episode ended. Total Reward: {cumulativeReward}");
                ResetEpisode();
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending reward/done: {e.Message}");
            isConnected = false;
        }
    }

    // Discrete action mapping:
    // 1: Forward  
    // 2: Forward + small left turn  
    // 3: Forward + large left turn  
    // 4: Forward + small right turn  
    // 5: Forward + large right turn
    void ApplyAction(int actionCode)
    {
        Debug.Log($"Applying action code: {actionCode}");
        switch (actionCode)
        {
            case 1:
                MoveForward();
                break;
            case 2:
                MoveForwardAndTurnLeft(smallTurnAngle);
                break;
            case 3:
                MoveForwardAndTurnLeft(largeTurnAngle);
                break;
            case 4:
                MoveForwardAndTurnRight(smallTurnAngle);
                break;
            case 5:
                MoveForwardAndTurnRight(largeTurnAngle);
                break;
            default:
                Debug.LogWarning("Unknown action code, defaulting to pure forward.");
                MoveForward();
                break;
        }
    }

    // Movement functions
    void MoveForward()
    {
        Debug.Log("Moving forward");
        agentTransform.Translate(Vector3.forward * moveSpeed * frameInterval, Space.Self);
        AddPositionToPath(agentTransform.position);
    }

    void MoveForwardAndTurnLeft(float angleDegrees)
    {
        Debug.Log("Moving forward with left turn (" + angleDegrees + "°)");
        float rotationAngle = Mathf.Min(angleDegrees, turnSpeed * frameInterval);
        agentTransform.Rotate(0f, -rotationAngle, 0f, Space.Self);
        agentTransform.Translate(Vector3.forward * moveSpeed * frameInterval, Space.Self);
        AddPositionToPath(agentTransform.position);
    }

    void MoveForwardAndTurnRight(float angleDegrees)
    {
        Debug.Log("Moving forward with right turn (" + angleDegrees + "°)");
        float rotationAngle = Mathf.Min(angleDegrees, turnSpeed * frameInterval);
        agentTransform.Rotate(0f, rotationAngle, 0f, Space.Self);
        agentTransform.Translate(Vector3.forward * moveSpeed * frameInterval, Space.Self);
        AddPositionToPath(agentTransform.position);
    }

    float GetCurrentSpeed() => moveSpeed;

    // ------------------------------------------------------------------
    // Helper method: Check if current state is terminal
    // ------------------------------------------------------------------
    bool IsTerminalState()
    {
        return !IsWithinBounds(agentTransform.position) ||
               collisionOccurred ||
               (collision_flag == 1.0f) ||
               targetReached ||
               timeLimitReached;
    }

    // ------------------------------------------------------------------
    // Refactored Reward Function
    // ------------------------------------------------------------------
    float GetReward()
    {
        bool terminal = IsTerminalState();
        float distanceToTarget = Vector3.Distance(agentTransform.position, target.position);
        float currentPotential = -distanceToTarget;
        float shapingReward = currentPotential - lastPotential; // potential-based shaping term
        lastPotential = currentPotential;

        float progressReward = (lastDistance - distanceToTarget) * progressScale;
        Vector3 toTarget = (target.position - agentTransform.position).normalized;
        float angleToTarget = Vector3.Angle(agentTransform.forward, toTarget);
        float headingPenalty = -(angleToTarget / 180f) * headingPenaltyScale;
        float timePenaltyReward = timePenalty;

        float currentSteeringAngle = agentTransform.eulerAngles.y;
        float steeringChange = Mathf.Abs(currentSteeringAngle - lastSteeringAngle);
        lastSteeringAngle = currentSteeringAngle;
        float smoothnessPenalty = -smoothnessPenaltyScale * steeringChange;

        // Combine all intermediate components
        float intermediateReward = progressReward + headingPenalty + shapingReward + timePenaltyReward + smoothnessPenalty;

        if (!terminal)
        {
            return intermediateReward;
        }

        // Terminal step: add bonuses/penalties as appropriate
        float finalReward = intermediateReward;
        if (targetReached && !timeLimitReached)
        {
            finalReward += targetBonus;
        }
        else if (usePartialSuccessThreshold && distanceToTarget < partialSuccessThreshold)
        {
            finalReward += targetBonus * partialSuccessFraction;
        }

        if (!targetReached && (stepCounter >= maxSteps || timeLimitReached))
        {
            float initialDistance = Vector3.Distance(initialPosition, target.position);
            float proximityFraction = 1f - Mathf.Clamp01(distanceToTarget / initialDistance);
            finalReward += targetBonus * 0.8f * proximityFraction;
        }

        if (collisionOccurred)
        {
            finalReward += collisionPenalty;
        }

        lastDistance = distanceToTarget;
        collision_flag = 0.0f;
        return finalReward;
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Obstacle") || collision.gameObject.CompareTag("Pedestrian"))
        {
            collisionOccurred = true;
            collision_flag = 1.0f;
            ResetAgent(false);
        }
    }

    // ------------------------------------------------------------------
    // Episode and Agent Reset Methods
    // ------------------------------------------------------------------
    void ResetAgent(bool timeLimitReset)
    {
        if (timeLimitReset)
            agentTransform.position = initialPosition + Vector3.right * 0.01f;
        else
            agentTransform.position = initialPosition;

        agentTransform.rotation = Quaternion.identity;
    }

    void ResetEpisode()
    {
        if (isResettingEpisode)
            return;
        isResettingEpisode = true;
        Debug.Log("Resetting episode");

        stepCounter = 0;
        collisionOccurred = false;
        targetReached = false;
        timeLimitReached = false;
        cumulativeReward = 0.0f;

        agentTransform.position = initialPosition;
        agentTransform.rotation = Quaternion.identity;
        lastDistance = Vector3.Distance(agentTransform.position, target.position);

        // Reset shaping parameters
        lastPotential = -lastDistance;
        lastSteeringAngle = agentTransform.eulerAngles.y;

        episodeStartTime = Time.time;
        StartNewPathLine();
        StartCoroutine(ResetCooldown());
    }

    IEnumerator ResetCooldown()
    {
        yield return new WaitForSeconds(0.2f);
        isResettingEpisode = false;
    }

    void AddPositionToPath(Vector3 position)
    {
        if (pathLineRenderer != null)
        {
            pathPositions.Add(position);
            pathLineRenderer.positionCount = pathPositions.Count;
            pathLineRenderer.SetPositions(pathPositions.ToArray());
        }
    }

    bool IsWithinBounds(Vector3 position)
    {
        float minX = Mathf.Min(p1.x, p2.x, p3.x, p4.x);
        float maxX = Mathf.Max(p1.x, p2.x, p3.x, p4.x);
        float minZ = Mathf.Min(p1.z, p2.z, p3.z, p4.z);
        float maxZ = Mathf.Max(p1.z, p2.z, p3.z, p4.z);

        return (position.x >= minX && position.x <= maxX &&
                position.z >= minZ && position.z <= maxZ);
    }

    void StartNewPathLine()
    {
        GameObject newLineObject = new GameObject("PathLine_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        newLineObject.layer = LayerMask.NameToLayer("PathLine");
        LineRenderer newLineRenderer = newLineObject.AddComponent<LineRenderer>();
        newLineRenderer.positionCount = 0;
        newLineRenderer.widthMultiplier = 0.2f;
        newLineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        newLineRenderer.useWorldSpace = true;

        Color newColor = new Color(
            UnityEngine.Random.Range(0.2f, 1.0f),
            UnityEngine.Random.Range(0.2f, 1.0f),
            UnityEngine.Random.Range(0.2f, 1.0f)
        );
        newLineRenderer.startColor = newColor;
        newLineRenderer.endColor = newColor;

        int pathLineLayerMask = 1 << LayerMask.NameToLayer("PathLine");
        screenshotCamera.cullingMask &= ~pathLineLayerMask;

        pathLineRenderer = newLineRenderer;
        pathPositions = new List<Vector3>();
        AddPositionToPath(agentTransform.position);
    }

    void OnApplicationQuit()
    {
        if (clientSocket != null)
        {
            if (clientSocket.Connected)
            {
                try { clientSocket.Shutdown(SocketShutdown.Both); }
                catch (Exception e) { Debug.LogWarning($"Socket shutdown error: {e.Message}"); }
            }
            clientSocket.Close();
        }
    }
}
