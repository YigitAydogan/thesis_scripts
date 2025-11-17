using System.Collections;
using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// Improved AI control script for NPCs that alternate between two target cubes,
/// moving to randomly generated destinations around each cube. The simulation step
/// is controlled by the 'simulationStep' variable, ensuring that NPC updates occur
/// at a fixed rate matching the main subject's simulation stepsize.
/// </summary>
public class AIControl : MonoBehaviour
{
    [Header("Navigation Settings")]
    [Tooltip("The radius around the target cube within which a random destination is chosen.")]
    public float destinationRadius = 5f;

    [Tooltip("Distance threshold to consider that the destination has been reached.")]
    public float destinationThreshold = 1f;

    [Header("Wait Time Settings")]
    [Tooltip("Minimum wait time upon reaching a destination.")]
    public float minWaitTime = 1f;

    [Tooltip("Maximum wait time upon reaching a destination.")]
    public float maxWaitTime = 3f;

    [Header("Simulation Settings")]
    [Tooltip("Simulation step duration (in seconds) to synchronize NPC updates with the subject.")]
    public float simulationStep = 0.2f;

    [Header("Animation Settings")]
    [Tooltip("Reference to the Animator component.")]
    public Animator animator;

    [Tooltip("Animator trigger for walking.")]
    public string walkingTrigger = "Walk";

    [Tooltip("Animator trigger for idle.")]
    public string idleTrigger = "Idle";

    private NavMeshAgent agent;
    private bool goingToCube1; // Indicates which cube is the current target
    private Vector3 currentDestination;

    // The target cubes will be retrieved from the TargetManager.
    private GameObject targetCube1;
    private GameObject targetCube2;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        if (agent == null)
        {
            Debug.LogError("NavMeshAgent component is missing from " + gameObject.name);
            enabled = false;
            return;
        }

        // Try to get the animator component automatically if not assigned.
        if (animator == null)
        {
            animator = GetComponent<Animator>();
            if (animator == null)
            {
                Debug.LogError("Animator component is missing from " + gameObject.name);
                enabled = false;
                return;
            }
        }

        // Retrieve target cubes from the centralized TargetManager.
        if (TargetManager.Instance != null)
        {
            targetCube1 = TargetManager.Instance.targetCube1;
            targetCube2 = TargetManager.Instance.targetCube2;
        }
        else
        {
            Debug.LogError("TargetManager instance not found in the scene!");
            enabled = false;
            return;
        }

        if (targetCube1 == null || targetCube2 == null)
        {
            Debug.LogError("Target cubes are not assigned in the TargetManager!");
            enabled = false;
            return;
        }

        // Randomly choose the initial target cube.
        goingToCube1 = (Random.value < 0.5f);

        // Start the behavior loop.
        StartCoroutine(StateMachine());
    }

    /// <summary>
    /// Main coroutine managing the NPC's behavior.
    /// This version uses WaitForSeconds(simulationStep) so that the NPC checks its state
    /// at a fixed rate (synchronized with the main subject's simulation stepsize).
    /// </summary>
    IEnumerator StateMachine()
    {
        while (true)
        {
            // Choose a new destination around the current target cube.
            SetNewDestination();

            // Trigger walking animation.
            animator.ResetTrigger(idleTrigger);
            animator.SetTrigger(walkingTrigger);

            // Wait until the NPC reaches the destination, checking every simulationStep seconds.
            while (!DestinationReached())
            {
                yield return new WaitForSeconds(simulationStep);
            }

            // Destination reached: stop moving and trigger idle animation.
            agent.ResetPath();
            animator.ResetTrigger(walkingTrigger);
            animator.SetTrigger(idleTrigger);

            // Wait for a random time at the destination.
            float waitTime = Random.Range(minWaitTime, maxWaitTime);
            yield return new WaitForSeconds(waitTime);

            // Switch target cube for the next destination.
            goingToCube1 = !goingToCube1;
        }
    }

    /// <summary>
    /// Sets a new destination by generating a random point around the current target cube.
    /// </summary>
    void SetNewDestination()
    {
        // Choose the current target based on goingToCube1.
        GameObject currentTarget = goingToCube1 ? targetCube1 : targetCube2;
        if (currentTarget == null)
        {
            Debug.LogError("Current target cube is not assigned!");
            return;
        }

        // Generate a random offset within a circle of the specified radius.
        Vector2 randomCircle = Random.insideUnitCircle * destinationRadius;
        Vector3 randomOffset = new Vector3(randomCircle.x, 0, randomCircle.y);

        // Calculate the destination position relative to the target cube.
        currentDestination = currentTarget.transform.position + randomOffset;

        // Ensure the destination is on the NavMesh.
        NavMeshHit hit;
        if (NavMesh.SamplePosition(currentDestination, out hit, destinationRadius, NavMesh.AllAreas))
        {
            currentDestination = hit.position;
        }

        agent.SetDestination(currentDestination);
    }

    /// <summary>
    /// Checks if the NPC has reached its current destination.
    /// </summary>
    /// <returns>True if destination is reached; otherwise, false.</returns>
    bool DestinationReached()
    {
        if (agent.pathPending)
            return false;

        // Check if the remaining distance is within the acceptable threshold.
        return agent.remainingDistance <= agent.stoppingDistance + destinationThreshold;
    }

    /// <summary>
    /// Visualizes the current destination in the Unity editor.
    /// </summary>
    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawSphere(currentDestination, 0.5f);
    }
}
