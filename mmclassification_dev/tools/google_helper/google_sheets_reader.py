from __future__ import print_function

import os.path
import random

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

class GoogleSheets(object):
    def __init__(self, spreadsheetId, node_rank=0):
        self.creds = None
        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                self.creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(self.creds.to_json())
            assert os.path.exists('token.json'), "Failed to initialize Google Sheets API. token.json not found."
            self.creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        self.service = build('sheets', 'v4', credentials=self.creds)
        self.sheet = self.service.spreadsheets()
        self.spreadsheetId = spreadsheetId
        self.task = None
        self.lock = None
        self.tab_name = None
        self.content_ptr = None
        self.node_rank = node_rank

    def read_data(self, tab_name, start_row="1", start_column="A", end_row=None, end_column=None):
        try:
            assert (end_row is not None and end_column is not None) or \
                   (end_row is None and end_column is None)
            assert start_column.isalpha() and start_row.isdigit()
            if end_row is not None:
                assert end_row.isdigit()
            if end_column is not None:
                assert end_column.isdigit()
            if end_row is not None and end_column is not None:
                read_range = "{}!{}{}:{}{}".format(tab_name, start_column, start_row, end_column, end_row)
            else:
                read_range = "{}!{}{}:Z1000".format(tab_name, start_column, start_row)
            # Call the Sheets API
            result = self.sheet.values().get(spreadsheetId=self.spreadsheetId,
                                             range=read_range).execute()
            values = result.get('values', [])
            return values
        except HttpError as err:
            # print(err)
            return -1

    def write_data(self, content, tab_name, start_row="1", start_column="A"):
        try:
            if type(start_column) is int or start_column.isdigit():
                start_column = chr(ord("A") + start_column - 1)
            rangeName = "{}!{}{}".format(tab_name, start_column, start_row)

            Body = {
                'values': content,
            }

            result = self.sheet.values().update(
                spreadsheetId=self.spreadsheetId, range=rangeName,
                valueInputOption='RAW', body=Body).execute()
            return 0
        except HttpError as err:
            # print(err)
            return -1

    def parse_head(self, head):
        start_column = 0
        for h in head:
            if h.startswith("cfg."):
                start_column += 1
            else:
                break
        assert start_column != 0, "Can not get valid config hyperparameters!"
        assert len(head) != start_column, "Invalid head format. len(head) == start_column"
        assert head[start_column] == "status", "Invalid head format. Expect head[{}] = 'status', but got '{}'".format(
            start_column, head[start_column]
        )
        return head[:start_column], start_column, start_column + 1

    def release_lock(self):
        if self.node_rank != 0:
            return
        if self.lock is not None and self.tab_name is not None:
            self.write_data([[-1]], self.tab_name, *self.lock)
            self.lock = None
        else:
            print("[WARNING] No lock to release.")

    def get_lock(self, lock_value):
        if self.node_rank != 0:
            return
        if self.lock is not None and self.tab_name is not None:
            self.write_data([[lock_value]], self.tab_name, *self.lock)
        else:
            print("[WARNING] No lock to get.")

    def process_lock(self):
        if self.node_rank != 0:
            return
        if self.lock is not None and self.tab_name is not None:
            self.write_data([[1]], self.tab_name, *self.lock)
        else:
            print("[WARNING] No lock to set process.")

    def finish_lock(self):
        if self.node_rank != 0:
            return
        if self.lock is not None and self.tab_name is not None:
            self.write_data([[2]], self.tab_name, *self.lock)
            self.lock = None
        else:
            print("[WARNING] No lock to set finish.")

    def update_result(self, content):
        if self.node_rank != 0:
            return
        self.write_data([content], self.tab_name, *self.content_ptr)
        self.finish_lock()

    def get_task(self, tab_name):
        full_content = self.read_data(tab_name)
        if type(full_content) is int:
            return {}, -1, -1, []
        head = full_content[1]
        cfg_kws, status_idx, content_idx = self.parse_head(head)
        tasks = full_content[2:]
        return cfg_kws, status_idx, content_idx, tasks

    def grab_task(self, tab_name, is_master=True, token=None):
        self.tab_name = tab_name
        task_dict = {}
        if not is_master:
            assert token is not None, "Token must be set for a slave node."
        if token is None:
            token = random.randint(10, 100000000)
        get_task_flag = False
        retry = 0
        while not get_task_flag and retry <= 20:
            cfg_kws, status_idx, content_idx, tasks = self.get_task(self.tab_name)
            if len(tasks) == 0:
                print("[Warning] Can not get any valid tasks.")
                retry += 1
            for row_idx, t in enumerate(tasks):
                if len(t) < status_idx:
                    continue
                if len(t) >= status_idx + 1:
                    try:
                        status = int(t[status_idx])
                    except:
                        continue
                else:
                    status = -1
                if status == token:
                    cfg_vals = t[:status_idx]
                    for cfg_kw, cfg_val in zip(cfg_kws, cfg_vals):
                        try:
                            cfg_val = float(cfg_val)
                        except:
                            pass
                        task_dict[cfg_kw] = cfg_val
                    self.content_ptr = [row_idx + 3, content_idx + 1]
                    get_task_flag = True
                    if is_master:
                        self.get_lock(0)
                    break
                elif status == -1 and is_master:
                    self.lock = [row_idx + 3, status_idx+1]
                    self.get_lock(token)
                    break
        if len(task_dict) == 0:
            print("[WARNING] Can not get valid task config.")
        return task_dict

    def __del__(self):
        self.release_lock()

if __name__ == '__main__':
    import random
    import time
    SAMPLE_SPREADSHEET_ID = '1znF0Bjncjk6SSrMOWxMSKstkTrusFg4ETkygLQgMyAc'
    SAMPLE_TAB_NAME = 'auto'
    gs = GoogleSheets(SAMPLE_SPREADSHEET_ID)
    while True:
        task_cfg = gs.grab_task(SAMPLE_TAB_NAME)
        if len(task_cfg) == 0:
            break
        print("Get task: {}".format(task_cfg))
        sleep_sec = random.randint(1, 20)
        print("Let us sleep {} sec...".format(sleep_sec))
        gs.process_lock()
        time.sleep(sleep_sec)
        res = [random.random() for _ in range(4)]
        print("Task {} finished with result {}".format(task_cfg, res))
        gs.update_result(res)
    print("All tasks have been completed, exit.")
