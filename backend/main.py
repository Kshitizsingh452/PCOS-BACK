import uvicorn
from os import getenv

if __name__=="__main__":
    port=int(getenv("PORT",9000))
    uvicorn.run("api.index:app",host="0.0.0.0",port=port,reload=True)
