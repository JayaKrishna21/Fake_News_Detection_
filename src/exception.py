import sys

from src.logger import logging

'''
The sys module in Python provides various functions and variables,
that are used to manipulate different parts of the Python runtime environment.
'''
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() #exc_tb is a total package which contains the info about the error, i.e., error file,line,type or message
    file_name = exc_tb.tb_frame.f_code.co_filename # gives the file_name where the error is encountered
    error_message = "Error occured in python script name [{0}] line number [{1}] with error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    return error_message


class CustomException(Exception):
    def __init__(self, error_message,error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail )

    def __str__(self):
        return self.error_message


# if __name__ == "__main__":

#     try:

#         a = b
        
#         '''
#         Example to create an exception to check whether the execption is being updated in the lgo files or not
#         For that,ZerDivisionError is encountered to raise an exception
#         '''

#     except Exception as e:
#         logging.info(e)
#         raise CustomException (e,sys)