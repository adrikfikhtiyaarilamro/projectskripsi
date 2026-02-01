using TCPServer;

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Runtime.InteropServices;
using Gtk;

class TcpServerApp
{
    // Windows API untuk keyboard simulation
    [DllImport("user32.dll")]
    static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
    
    [DllImport("user32.dll", SetLastError = true)]
    static extern IntPtr FindWindow(string? lpClassName, string lpWindowName);
    
    [DllImport("user32.dll")]
    static extern bool SetForegroundWindow(IntPtr hWnd);
    
    [DllImport("user32.dll")]
    static extern IntPtr GetForegroundWindow();

    const int KEYEVENTF_KEYDOWN = 0x0000;
    const int KEYEVENTF_KEYUP = 0x0002;
    
    // Virtual Key Codes
    const byte VK_LEFT = 0x25;
    const byte VK_UP = 0x26;
    const byte VK_RIGHT = 0x27;
    const byte VK_DOWN = 0x28;
    const byte VK_SPACE = 0x20;
    const byte VK_W = 0x57;
    const byte VK_A = 0x41;
    const byte VK_S = 0x53;
    const byte VK_D = 0x44;
    
    static IntPtr gameWindowHandle = IntPtr.Zero;

    static TcpListener? listener;
    static bool isRunning = false;
    static TextView? logTextView;
    static Thread? serverThread;
    static DrawingArea? canvasArea;

    static Kotak box = new Kotak(50, 20, 20);

    public static void Main()
    {
        Application.Init();

        // === Window Setup ===
        Window win = new Window("TCP Server GTK");
        win.SetDefaultSize(400, 300);
        win.DeleteEvent += delegate { isRunning = false; listener?.Stop(); Application.Quit(); };

        VBox vbox = new VBox(false, 5);

        Button startBtn = new Button("Start Server");
        Button stopBtn = new Button("Stop Server");
        logTextView = new TextView();
        logTextView.Editable = false;

        // === Drawing Area (Canvas) ===
        canvasArea = new DrawingArea();
        canvasArea.SetSizeRequest(400, 150);
        canvasArea.Drawn += OnDrawCanvas;

        vbox.PackStart(startBtn, false, false, 0);
        vbox.PackStart(stopBtn, false, false, 0);
        vbox.PackStart(canvasArea, false, false, 5);
        ScrolledWindow scrolled = new ScrolledWindow();
        scrolled.Add(logTextView);
        vbox.PackStart(scrolled, true, true, 0);


        win.Add(vbox);
        win.ShowAll();

        // === Button Events ===
        startBtn.Clicked += (s, e) =>
        {
            if (!isRunning)
            {
                isRunning = true;
                serverThread = new Thread(StartServer);
                serverThread.Start();
                AppendLog("Server started.");
            }
        };

        stopBtn.Clicked += (s, e) =>
        {
            isRunning = false;
            listener?.Stop();
            AppendLog("Server stopped.");
        };

        Application.Run();
    }

    static void OnDrawCanvas(object o, DrawnArgs args)
    {
        var cr = args.Cr;
        var allocation = canvasArea?.Allocation;

        if (allocation == null){
            return;
        }

        cr.SetSourceRGB(0, 1, 0);
        cr.Rectangle(box.X, box.Y, box.Width, box.Height);
        cr.Fill();
    }

    static void StartServer()
    {
        int port = 5005;
        listener = new TcpListener(IPAddress.Any, port);
        listener.Start();

        while (isRunning)
        {
            try
            {
                TcpClient client = listener.AcceptTcpClient();
                AppendLog("Client connected.");
                Thread t = new Thread(() => HandleClient(client));
                t.Start();
            }
            catch
            {
                if (!isRunning) break;
            }
        }
    }

    static void HandleClient(TcpClient client)
    {
        NetworkStream stream = client.GetStream();
        byte[] buffer = new byte[1024];

        try
        {
            while (true)
            {
                int byteCount = stream.Read(buffer, 0, buffer.Length);
                if (byteCount == 0) break;
                string message = Encoding.UTF8.GetString(buffer, 0, byteCount).Trim().ToLower();
                AppendLog($"🎤 Received: {message.ToUpper()}");

                if (message == "left" || message == "right" || message == "up" || message == "down")
                {
                    box.Move(message);  // Gerakkan kotak
                    Application.Invoke(delegate {
                        canvasArea?.QueueDraw();
                    }); // Gambar ulang
                    
                    // ⭐ SEND KEYBOARD TO GAME
                    SendKeyToGame(message);

                    // ⭐ Send ACK back to client with server timestamp & game status
                    try
                    {
                        long serverTsMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                        string gameStatus = box.LastMovementStatus; // "success" atau "blocked"
                        string ack = $"ACK|{message}|{serverTsMs}|{gameStatus}\n";
                        byte[] ackBytes = Encoding.UTF8.GetBytes(ack);
                        stream.Write(ackBytes, 0, ackBytes.Length);
                        AppendLog($"💬 ACK sent: {ack.Trim()} | Game: {gameStatus}");
                    }
                    catch (Exception ex)
                    {
                        AppendLog($"ACK error: {ex.Message}");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            AppendLog("Error: " + ex.Message);
        }
        finally
        {
            client.Close();
            AppendLog("Client disconnected.");
        }
    }

    static void AppendLog(string message)
	{
	    Application.Invoke(delegate
	    {
	        if (logTextView == null)
	        {
	            Console.WriteLine("logTextView is not initialized.");
	            return;
	        }

	        var buffer = logTextView.Buffer;
	        string timestamped = $"{DateTime.Now:HH:mm:ss} - {message}\n";
	        
	        // Sisipkan log baru di awal teks
	        buffer.Text = timestamped + buffer.Text;

	        // Auto-scroll ke atas (karena log terbaru ada di atas)
	        TextIter startIter = buffer.StartIter;
	        logTextView.ScrollToIter(startIter, 0, false, 0, 0);

            canvasArea?.QueueDraw();
	    });
	}

    static void SendKeyToGame(string command)
    {
        byte key = 0;
        string keyName = "";

        // Map voice commands to keyboard keys
        switch (command)
        {
            case "left":
                key = VK_A;  // Ganti ke WASD
                keyName = "A";
                break;
            case "right":
                key = VK_D; // Ganti ke WASD
                keyName = "D";
                break;
            case "up":
                key = VK_W; // W untuk jump
                keyName = "W";
                break;
            case "down":
                key = VK_S;  // S untuk crouch
                keyName = "S";
                break;
        }

        if (key != 0)
        {
            // Coba beberapa window title Unity
            if (gameWindowHandle == IntPtr.Zero)
            {
                string[] possibleTitles = { 
                    "Aing Kasep", 
                    "Endless Runner",
                    "Unity",
                    "Game 3D Endless Runner"
                };
                
                foreach (var title in possibleTitles)
                {
                    gameWindowHandle = FindWindow(null, title);
                    if (gameWindowHandle != IntPtr.Zero)
                    {
                        AppendLog($"✓ Found game: '{title}'");
                        break;
                    }
                }
                
                if (gameWindowHandle == IntPtr.Zero)
                {
                    AppendLog("⚠️ Game window not found - sending to active window");
                }
            }
            
            // Focus ke game window jika ditemukan
            if (gameWindowHandle != IntPtr.Zero)
            {
                SetForegroundWindow(gameWindowHandle);
                Thread.Sleep(50); // Tunggu focus
            }
            
            // Simulate key press (press 2x untuk lebih reliable)
            keybd_event(key, 0, KEYEVENTF_KEYDOWN, UIntPtr.Zero);
            Thread.Sleep(150); // Hold longer
            keybd_event(key, 0, KEYEVENTF_KEYUP, UIntPtr.Zero);
            
            AppendLog($"⌨️  Sent: {keyName} | Cmd: {command}");
        }
    }

}
