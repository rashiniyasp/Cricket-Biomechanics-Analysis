class PhaseDetector:
    def __init__(self):
        self.current_phase = "Stance"
        self.colors = {
            "Stance": (200, 200, 200),   # Silver
            "Trigger": (0, 255, 255),    # Yellow
            "Execution": (0, 0, 255)     # Red
        }

    def detect_phase(self, knee_angle, head_deviation):
        """
        One-Way State Machine:
        Stance -> Trigger -> Execution
        (Prevents flickering back and forth)
        """
        
        # 1. If we are in STANCE, check if we should upgrade to TRIGGER
        if self.current_phase == "Stance":
            # If head moves significantly (>15px), he has started moving
            if head_deviation > 15:
                self.current_phase = "Trigger"

        # 2. If we are in TRIGGER, check if we should upgrade to EXECUTION
        elif self.current_phase == "Trigger":
            # If Knee compresses (< 155) OR Head moves far (> 80), it's the shot
            if knee_angle < 155 or head_deviation > 80:
                self.current_phase = "Execution"

        # 3. If in EXECUTION, stay there (Lock the state)
        # This prevents the label from flickering back to 'Trigger' during the follow-through
        
        return self.current_phase, self.colors[self.current_phase]