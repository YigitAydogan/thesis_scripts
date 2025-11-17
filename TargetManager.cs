using UnityEngine;

/// <summary>
/// A centralized manager that holds references to the target cubes.
/// Assign these references once in the Inspector and all NPCs will use them.
/// </summary>
public class TargetManager : MonoBehaviour
{
    public static TargetManager Instance { get; private set; }

    [Tooltip("Reference to the first target cube (e.g., Entrance A).")]
    public GameObject targetCube1;

    [Tooltip("Reference to the second target cube (e.g., Entrance B).")]
    public GameObject targetCube2;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            // Optional: Uncomment if you want the manager to persist across scenes.
            // DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
}
