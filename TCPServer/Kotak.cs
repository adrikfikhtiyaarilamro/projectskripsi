namespace TCPServer
{
    public class Kotak
    {
        private readonly double[] lanes = { 100, 200, 300 }; // Left, Center, Right
        private int currentLaneIndex = 1; // start in center lane (lanes[1])

        public double X => lanes[currentLaneIndex];
        public double Y { get; private set; }
        public double Width { get; set; }
        public double Height { get; set; }
        public string LastMovementStatus { get; private set; } = "idle"; // Track last move result
        public string LastMovement { get; private set; } = ""; // Track last successful move

        private double groundY;
        private double jumpHeight = 30;
        private double slideHeight = 30;

        public Kotak(double y, double width = 50, double height = 50)
        {
            groundY = y;
            Y = y;
            Width = width;
            Height = height;
        }

        public void Move(string direction)
        {
            bool moved = false;
            switch (direction.ToLower())
            {
                case "left":
                    if (currentLaneIndex > 0)
                    {
                        currentLaneIndex--;
                        moved = true;
                    }
                    break;

                case "right":
                    if (currentLaneIndex < lanes.Length - 1)
                    {
                        currentLaneIndex++;
                        moved = true;
                    }
                    break;

                case "up": // jump
                    Y = groundY - jumpHeight;
                    moved = true;
                    break;

                case "down": // slide
                    Y = groundY + slideHeight;
                    moved = true;
                    break;
            }

            // Track movement status
            LastMovement = direction.ToLower();
            LastMovementStatus = moved ? "success" : "blocked"; // success/blocked (e.g., already at edge)

            // Kembalikan ke posisi semula setelah waktu singkat (simulate jump/slide)
            var timer = new System.Timers.Timer(300); // 300ms
            timer.Elapsed += (s, e) =>
            {
                Y = groundY;
                timer.Stop();
                timer.Dispose();
            };
            timer.Start();
        }
    }
}
