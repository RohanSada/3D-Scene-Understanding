def controller_process(ctrl_queue):
    while True:
        msg = ctrl_queue.get()
        print("[CTRL]", msg["MsgType"])