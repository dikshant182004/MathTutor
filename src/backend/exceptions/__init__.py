import sys


def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "<unknown>"
    line_no = exc_tb.tb_lineno if exc_tb else -1
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        line_no,
        str(error),
    )
    return error_message


class Agent_Exception(Exception):
    def __init__(self, error, error_detail: sys):
        self.error_message = error_message_detail(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message