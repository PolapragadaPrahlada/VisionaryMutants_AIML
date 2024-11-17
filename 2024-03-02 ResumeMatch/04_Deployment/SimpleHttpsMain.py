import uvicorn
import ssl

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain("cert.pem", keyfile="key.pem")

#--------------------------------------------------------------------------------#
# Main Code.
#--------------------------------------------------------------------------------#
if __name__ == "__main__":
    uvicorn.run("SimpleHttps:App", host="0.0.0.0", port=8000, ssl=ssl_context)
