
class UserInput:
    def __init__(self, data_start='2018-02-01', data_end='2020-01-31',
                 pred_start='2020-02-01', pred_end='2020-03-05',
                 drive_path='./', area=None, is_crawling=True, is_modelfromfile=False, is_tune_para=True):
        self.data_start = data_start
        self.data_end = data_end

        self.pred_start = pred_start
        self.pred_end = pred_end

        # 관측소 선택
        self.area = area

        self.is_crawling = is_crawling
        self.is_modelfromfile = is_modelfromfile
        self.is_tune_para = is_tune_para

        # 파일 기본경로 지정
        self.drive_path = drive_path
        self.data_path = self.drive_path + 'data/'
        self.output_path = self.drive_path + 'output/'
