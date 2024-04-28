# from vidstream import StreamingServer, ScreenShareClient
# import threading

# def get_ip_address():
#     import socket
#     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     try:
#         s.connect(("8.8.8.8", 80))
#         ip_address = s.getsockname()[0]
#     except Exception as e:
#         print("Error:", e)
#         ip_address = None
#     finally:
#         s.close()
#     return ip_address

# # Get your IP address
# your_ip_address = get_ip_address()
# if your_ip_address:
#     print("Your IP address is:", your_ip_address)
# else:
#     print("Failed to retrieve IP address.")

# # Create a streaming server on localhost (127.0.0.1) and port 8001
# server = StreamingServer('127.0.0.1', 8001)

# # Start the server in a separate thread
# server_thread = threading.Thread(target=server.start_server)
# server_thread.start()

# # Create a screen sharing client
# client = ScreenShareClient('127.0.0.1', 8001)

# # Start streaming your desktop screen
# client.start_stream()


from vidstream import StreamingServer, ScreenShareClient
import threading

# Create a streaming server on localhost (127.0.0.1) and port 8001
server = StreamingServer('127.0.0.1', 8001)

# Start the server in a separate thread
server_thread = threading.Thread(target=server.start_server)
server_thread.start()

# Create a screen sharing client
client = ScreenShareClient('127.0.0.1', 8001)

# Start streaming your desktop screen
client.start_stream()