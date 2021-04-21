from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import MySQLdb as sql
import requests
from sklearn.preprocessing import MinMaxScaler

def symmetric_mean_absolute_percentage_error(pred, real):
    """
    Calculates sMAPE
    :param real: actual values
    :param pred: predicted values
    :return: sMAPE
    """
    a = np.reshape(real, (-1,))
    b = np.reshape(pred, (-1,))
    sMAPE = np.mean(np.abs(a - b) / (np.abs(a) + np.abs(b))*2).item()
    return round(sMAPE*100,2)


def connect_to_data_base(config):
    try:
        db = sql.connect(host=config.db_host, user=config.db_username, passwd=config.db_password, db=config.db_name,
                         port=config.db_port, charset='utf8')
    except:
        raise ConnectionError('Could not connect to database server, check internet connection and database detail..')
    return db


def login(config):
    ntels_login_url = config.ntels_host + config.login_url
    login_params = dict(
        user_id=config.ntels_username,
        user_pw=config.ntels_password
    )
    with requests.Session() as login_sess:
        try:
            login_sess.post(ntels_login_url, login_params, verify=False, allow_redirects=True, timeout=90)
            return login_sess
        except:
            raise ConnectionError('Could not login to server, check server configuration..')


def get_data_from_HTTPS_request(url, login_sess):
    '''
	get json data format from HTTP API
	:param url: http address
	:param params: parameters
	:return: json data format
	'''
    ### login first
    try:
        response = login_sess.get(url, verify=False, timeout=90)
        return response.json()
    except:
        raise ConnectionError('Could not get data from https request, check server API configuration')


def get_data_from_HTTPS_post_request(url, login_sess, params):
    '''
	get json data format from HTTP API
	:param url: http address
	:param params: parameters
	:return: json data format
	'''
    ### login first
    try:
        response = login_sess.post(url, data=params, verify=False, timeout=90)
        return response.json()
    except:
        raise ConnectionError('Could not get data from {}, check server API configuration'.format(url))


def load_energy_data_from_HTTPS_API(config, building_ids=None):
    ### get energy data using web API
    login_sess = login(config)
    host = config.ntels_host
    url = host + "/NISBCP/urbanmap/energy/getBuildingEnergyTrend.ajax?"
    pad = "&"
    start_time = config.start_time
    end_time = config.end_time
    period = "period=15m"
    data_list = []
    building_list = []
    if building_ids is None:  ### get data of all building
        db = connect_to_data_base(config)
        cursor = db.cursor()
        query = 'SELECT bld_id FROM nisbcp.t_buildings where isismart=\'Y\';'
        cursor.execute(query)
        building_ids = cursor.fetchall()
        cursor.close()
        db.close()
        ## getbuilding list
        building_ids = ["bid=" + bld_id[0].strip() for bld_id in building_ids]
    ##initialize data_list and building_list to hold data
    else:  ### get data of buildings that are in the list
        building_ids = ["bid=" + bld_id.strip() for bld_id in building_ids]
    for idx, building_id in enumerate(building_ids):
        url = url + start_time + pad + end_time + pad + building_id + pad + period
        json = get_data_from_HTTPS_request(url, login_sess)
        ##reset url
        url = host + "/NISBCP/urbanmap/energy/getBuildingEnergyTrend.ajax?"
        #### API data has a field to determind whether return data is empty or not.
        empty = json['empty']
        ## real data is stored in 'list' field
        json = json["list"]
        dates = []
        energy_values = []
        if empty == False:
            for log in json:
                dates.append(log["logdate"])
                energy_values.append(log["usage"])
            dates = pd.to_datetime(dates)
            data = pd.DataFrame(energy_values, index=dates, columns=["Power"])
            data_list.append(data)
            building_list.append(json[0]["bld_id"])
        else:
            print('Building {} does not have energy data from {} to {}'.format(building_id, start_time, end_time))
    return data_list, building_list


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs((y_true - y_pred) / y_true)) / len(y_pred)


def json_to_dataframe(path, col_name='Power', orient='index'):
    import json
    with open(path) as in_stream:
        data_dict = json.load(in_stream)
    data = pd.DataFrame.from_dict(data_dict, orient=orient, columns=[col_name])
    return data


##### export dataframe to json format  = {index : value}
def dataframe_to_dict(data):
    data_dict = {}
    for idx, row in data.iterrows():
        data_dict[str(idx)] = row[0]
    return data_dict


## export dictionary to json file format
def dict_to_json(folder, filename, dict):
    '''
	dict must be a 1 level dictionary )not contains list
	:param folder:
	:param filename:
	:param dict:
	:return:
	'''
    import json
    import os
    ## create 'analyzed' folder (if neccessary)
    if folder != '':
        os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), 'w') as out_stream:
        json.dump(dict, out_stream)


### export dataframe to csv file format
def dataframe_to_csv(folder, filename, dataframe):
    import os
    if folder != '':
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
    else:
        path = filename
    try:
        dataframe.to_csv(path)
    except:
        raise IOError('Cannot create file, check permission..')


def loadData(path, idx_col=0, freq='H', formula='mean'):
    '''
	load data from file in path
	:param path: file path
	:param idx_col: index of indexing column
	:return: dataframe + building name
	'''
    try:
        if path.endswith('.csv'):
            data = pd.read_csv(path)
            building = path.split('\\')[-1].split('/')[-1].replace('.csv', '')
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            data = pd.read_excel(path)
            building = path.split('\\')[-1].split('/')[-1].replace('.xlsx', '').replace('.xls', '')
        elif path.endswith('.json'):
            data = json_to_dataframe(path)
            building = path.split('\\')[-1].split('/')[-1].replace('.json', '')
        else:
            print('Not supported file format..')
            return
    except:
        raise Exception('Cannot read files format from input folder..')
    ## set data index
    datetimeColName = data.columns[idx_col]
    data[datetimeColName] = pd.to_datetime(data[datetimeColName])
    data.set_index(datetimeColName, inplace=True)
    ## sort data by index
    data = data.sort_index(axis=0)
    ### checking for non-numertic value and convert using interpolate function
    for i in data.columns:
        data[i] = pd.to_numeric(data[i], errors='coerce').interpolate(method='quadratic')
    ##### fill missing data with interpolate value
    # (suppose that 15 minutes is minimum data collected interval.
    # 1 minutes make function performance slow down)
    # if data is collected using smaller interval, decrease frequence param
    data = data.resample('15T').interpolate()
    data = data.reindex(pd.date_range(data.index[0], data.index[-1], freq=freq)).fillna(method='bfill')
    ### resample data to freq regulation
    if formula == 'sum':  ## resampling using sum function
        ### resample data to fill missing value. then reindex to 15 frequence
        data = data.resample(freq).sum()
    elif formula == 'mean':  ## resampling using mean() function
        data = data.resample(freq).mean()
    return data, building


def add_weekday_column(data):
    '''
	detect holiday and add a categorical column to dataframe
	:param data: original data
	:return: appended data
	'''
    from workalendar.asia import SouthKorea
    index = data.index
    ko_calendar = SouthKorea()
    is_holiday = []
    for time_ in index:
        if ko_calendar.is_working_day(time_.date()):
            is_holiday.append(time_.date().weekday())
        else:
            if time_.date().weekday() in [5, 6]:
                is_holiday.append(time_.date().weekday())
            else:
                is_holiday.append(7)
    is_holiday = pd.DataFrame(is_holiday, columns=['weekday_index'], index=index)
    appended_data = pd.concat([data, is_holiday], axis=1)
    return appended_data



def get_weekday_index(date):
    from workalendar.asia import SouthKorea
    ko_calendar = SouthKorea()
    if ko_calendar.is_working_day(date.date()):
        return (date.date().weekday())
    else:
        if date.date().weekday() in [5, 6]:
            return(date.date().weekday())
        else:
            return(7)



def scale_to_0and1(data):
    scaler = MinMaxScaler(copy=False,feature_range=(0, 1))  # MinMax Scaler
    data = scaler.fit_transform(data)  # input: ndarray type data
    return (data, scaler)


# split into train and test sets
def splitData(data, trainPercent=0.8):
    train_size = int(len(data) * trainPercent)
    train_data, test_data = data.iloc[0:train_size, ], data.iloc[train_size:, ]
    print("\n", "train length:", len(train_data), "\n", "test length:", len(test_data))
    return (train_data, test_data)


def get_data_filenames_from_folder(path, ext='.csv'):
    import os
    try:
        filenames = [os.path.join(path, file) for file in os.listdir(path) if
                     os.path.isdir(os.path.join(path, file)) == False]
        filenames = sorted(filenames)
        data_files = []
        for file in filenames:
            if file.endswith(ext):
                data_files.append(file)
    except FileNotFoundError:
        raise FileNotFoundError('Cannot locate the specific folder..')
    return data_files


def get_data_by_time(energy_data, start_time=None, end_time=None, freq='H'):
    if start_time is None and end_time is None:
        start_time = energy_data.index[0]
        end_time = energy_data.index[-1]
        try:
            start_time, end_time = pd.to_datetime(start_time), pd.to_datetime(end_time)
            start_time, end_time = str(start_time.date()), str(end_time.date())
        except ValueError as e:
            raise (e)
    else:
        ## if one of start or end time is None
        start_time = end_time if start_time is None else start_time
        end_time = start_time if end_time is None else end_time
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        ### if user input time with format YYYY/MM/DD --> set time for start and end time
        if start_time.time().hour == start_time.time().minute == 0:
            if freq == 'H':
                start_time = start_time.replace(hour=0)
                end_time = end_time.replace(hour=23)
            elif freq == '15T' or freq == '15m':
                start_time = start_time.replace(hour=0, minute=0)
                end_time = end_time.replace(hour=23, minute=45)
        else:  ## if user input time with format YYYY/MM/DD HH:MM --> set time for end time corresponding to start time
            if freq == 'H':
                start_time = start_time.replace(minute=0)
                end_time = end_time.replace(day=start_time.date().day + 1, hour=start_time.time().hour,
                                            minute=start_time.time().minute)
            elif freq == '15T' or freq == '15m':
                start_time = start_time.replace(minute=start_time.time().minute - (
                        start_time.time().minute % 15))  ### round up minute to 15 minute freq
                end_time = end_time.replace(day=start_time.date().day + 1, hour=start_time.time().hour,
                                            minute=start_time.time().minute - (
                                                    start_time.time().minute % 15))  ### round up minute to 15 minute freq
    if start_time > end_time:
        raise ValueError('Start time must be a time before end time..')
    elif start_time <= end_time:
        if start_time not in energy_data.index or end_time not in energy_data.index:
            print('start time {}'.format(start_time))
            print('end_time  {}'.format(end_time))
            print(energy_data.index)
            raise ValueError('Time range exceed data range, please choose different time range..')
        else:
            period_data = energy_data[start_time:end_time]
            return period_data


def get_time_by_month(month, year, freq='H'):
    import datetime
    today = datetime.date.today()
    if month is None or int(month) > 12 or int(month) < 1:
        raise Exception('Unreconized month, make sure month is integer from 1-12..')
    elif year is None or int(year) < 1970 or int(year) > datetime.date.today().year:
        raise Exception('Unreconized year, make sure passed year is integer from 1970 to this year..')
    else:
        ### determine month first and last day
        if year == today.year:
            if month == today.month:
                start_time = today.replace(day=1)
                start_time = pd.to_datetime(start_time)
                end_time = pd.datetime.now()
                if freq == 'H':
                    end_time = end_time.replace(minute=0, second=0, microsecond=0)
                elif freq == '15T' or freq == '15m':
                    end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
                                                microsecond=0)
                end_time = pd.to_datetime(end_time)
                return start_time, end_time
            elif month < today.month:
                latter_month_first_day = today.replace(month=month + 1, day=1)
                month_last_day = latter_month_first_day - datetime.timedelta(days=1)
                month_first_day = month_last_day.replace(day=1)
                ## convert date to date time
                start_time = pd.to_datetime(month_first_day)
                end_time = pd.to_datetime(month_last_day)
                ## edit end_time due to frequence
                if freq == 'H':
                    end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
                elif freq == '15T' or freq == '15m':
                    end_time = end_time.replace(hour=23, minute=45,
                                                second=0, microsecond=0)
                return start_time, end_time
            else:
                raise ValueError('Month is not valid..')
        else:
            if month == 12:
                latter_year_first_day = today.replace(year=year + 1, month=1, day=1)
                last_day = latter_year_first_day - datetime.timedelta(days=1)
                first_day = last_day.replace(day=1)
                ## convert date to date time
                start_time = pd.to_datetime(first_day)
                end_time = pd.to_datetime(last_day)
                ## edit end_time due to frequence
                if freq == 'H':
                    end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
                elif freq == '15T' or freq == '15m':
                    end_time = end_time.replace(hour=23, minute=45,
                                                second=0, microsecond=0)
                return start_time, end_time
            else:
                latter_month_first_day = today.replace(year=year, month=month + 1, day=1)
                month_last_day = latter_month_first_day - datetime.timedelta(days=1)
                month_first_day = month_last_day.replace(day=1)
                ## convert date to date time
                start_time = pd.to_datetime(month_first_day)
                end_time = pd.to_datetime(month_last_day)
                ## edit end_time due to frequence
                if freq == 'H':
                    end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
                elif freq == '15T' or freq == '15m':
                    end_time = end_time.replace(hour=23, minute=45,
                                                second=0, microsecond=0)
                return start_time, end_time


def get_time_by_keyword(keyword='today', freq='H'):
    # keyword in ['today', 'yesterday', 'this week', 'last week', 'this month', 'last month', 'this year', 'last year']
    import datetime
    keyword = keyword.lower().strip()
    assert keyword in ['tomorow', 'today', 'yesterday', 'this week', 'last week', 'this month', 'last month',
                       'this year',
                       'last year']
    if keyword == 'today':
        today = pd.datetime.today().date()
        start_time = pd.to_datetime(today)
        end_time = pd.datetime.now()
        if freq == 'H':
            end_time = end_time.replace(minute=0, second=0, microsecond=0)
        elif freq == '15T' or freq == '15m':
            end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
                                        microsecond=0)
        end_time = pd.to_datetime(end_time)
        return start_time, end_time
    elif keyword == 'tomorow':
        tomorow = datetime.date.today() + datetime.timedelta(days=1)
        start_time = pd.to_datetime(tomorow)
        end_time = start_time
        if freq == 'H':
            end_time = start_time.replace(hour=23, minute=0, second=0, microsecond=0)
        else:
            end_time = start_time.replace(hour=23, minute=45,
                                          second=0, microsecond=0)
        return start_time, end_time
    elif keyword == 'yesterday':
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        start_time = pd.to_datetime(yesterday)
        end_time = start_time
        if freq == 'H':
            end_time = start_time.replace(hour=23, minute=0, second=0, microsecond=0)
        else:
            end_time = start_time.replace(hour=23, minute=45,
                                          second=0, microsecond=0)
        return start_time, end_time
    elif keyword == 'this week':
        today = datetime.date.today()
        start_time = today - datetime.timedelta(days=today.weekday())
        start_time = pd.to_datetime(start_time)
        end_time = pd.datetime.now()
        if freq == 'H':
            end_time = end_time.replace(minute=0, second=0, microsecond=0)
        elif freq == '15T' or freq == '15m':
            end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
                                        microsecond=0)
        end_time = pd.to_datetime(end_time)
        return start_time, end_time
    elif keyword == 'last week':
        today = datetime.date.today()
        start_time = today - datetime.timedelta(days=today.weekday() + 7)
        start_time = pd.to_datetime(start_time)
        end_time = start_time + datetime.timedelta(days=6)
        if freq == 'H':
            end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
        elif freq == '15T' or freq == '15m':
            end_time = end_time.replace(hour=23, minute=45,
                                        second=0, microsecond=0)
        end_time = pd.to_datetime(end_time)
        return start_time, end_time
    elif keyword == 'this month':
        today = datetime.date.today()
        this_month_first_day = today.replace(day=1)
        start_time = pd.to_datetime(this_month_first_day)
        end_time = pd.datetime.now()
        if freq == 'H':
            end_time = end_time.replace(minute=0, second=0, microsecond=0)
        elif freq == '15T' or freq == '15m':
            end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
                                        microsecond=0)
        end_time = pd.to_datetime(end_time)
        return start_time, end_time
    elif keyword == 'last month':
        today = datetime.date.today()
        this_month_first_day = today.replace(day=1)
        last_month_last_day = this_month_first_day - datetime.timedelta(days=1)
        last_month_first_day = last_month_last_day.replace(day=1)
        start_time, end_time = last_month_first_day, last_month_last_day
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        if freq == 'H':
            end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
        elif freq == '15T' or freq == '15m':
            end_time = end_time.replace(hour=23, minute=45,
                                        second=0, microsecond=0)
        return start_time, end_time
    elif keyword == 'this year':
        today = datetime.date.today()
        this_year_first_day = today.replace(month=1, day=1)
        start_time = pd.to_datetime(this_year_first_day)
        end_time = pd.datetime.now()
        if freq == 'H':
            end_time = end_time.replace(minute=0, second=0, microsecond=0)
        elif freq == '15T' or freq == '15m':
            end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
                                        microsecond=0)
        end_time = pd.to_datetime(end_time)
        return start_time, end_time
    elif keyword == 'last year':
        today = datetime.date.today()
        last_year_first_day = today.replace(year=(today.year - 1), month=1, day=1)
        last_year_last_day = today.replace(year=(today.year - 1), month=12, day=31)
        start_time = last_year_first_day
        end_time = last_year_last_day
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        if freq == 'H':
            end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
        elif freq == '15T' or freq == '15m':
            end_time = end_time.replace(hour=23, minute=45,
                                        second=0, microsecond=0)
        return start_time, end_time
    else:
        print('Unknown keyword..')
        return


def load_weather_data_from_database(config, start_time, end_time):
    ### get weather data from database
    db = connect_to_data_base(config)
    cursor = db.cursor()
    query = 'SELECT wdate, temperature, humidity FROM nisbcp.t_weather_data where wdate between "' + start_time + '" and "' + end_time + '"'
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        rows = [list(row) for row in data]  ## read row from query
        ## create dataframe and put data into
        weather_data = pd.DataFrame(rows, columns=['Time', 'Temperature', 'Humidity'])
        ## set index
        weather_data['Time'] = pd.to_datetime(weather_data['Time'])
        weather_data.set_index('Time', inplace=True)
        ## close database connection
        cursor.close()
        db.close()
    except:
        cursor.close()
        db.close()
        raise Exception('Could not execute the query, check statement or connection to database..')
    return weather_data


def get_building_ids(config, group_id=None):
    db = connect_to_data_base(config)
    cursor = db.cursor()
    if group_id is not None:
        query = 'select bld_id from nisbcp.l_groups_bld where grp_id=\'' + str(group_id) + '\';'
        try:
            cursor.execute(query)
            data = cursor.fetchall()
            building_list = [bld_id[0] for bld_id in data]
            cursor.close()
            db.close()
        except:
            cursor.close()
            db.close()
            raise Exception('Could not execute the query, check statement or connection to database..')
        return building_list
    else:
        query = 'SELECT bld_id FROM nisbcp.t_buildings where isismart=\'Y\';'
        cursor.execute(query)
        building_ids = cursor.fetchall()
        cursor.close()
        db.close()
        building_list = [bld_id[0].strip() for bld_id in building_ids]
        return building_list


def get_group_ids(config):
    db = connect_to_data_base(config)
    cursor = db.cursor()
    query = 'select grp_id, grp_name from nisbcp.l_groups;'
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        group_ids = [grp_id[0] for grp_id in data]
        group_names = [grp_name[1] for grp_name in data]
        cursor.close()
        db.close()
    except:
        cursor.close()
        db.close()
        raise Exception('Could not execute the query, check statement or connection to database..')
    return group_ids, group_names


def get_dust_data_from_other_source():
    import requests
    param_avg = dict(
        serviceKey='a2khNs3hR3z2Xx4gSeXCI0qA2dhIKkMOtVhLLQGERAT9aB4j3bHUqXPII7MW+493KZ50nPr164PbSpj5LayUlg==',
        numOfRows='1000',
        pageSize='100',
        pageNo='1',
        startPage='1',
        itemCode='PM10',
        dataGubun='HOUR',
        searchCondition='YEAR',
        _returnType='json',
        ver='1.3',
    )
    url_avg = 'http://openapi.airkorea.or.kr/openapi/services/rest/ArpltnInforInqireSvc/getCtprvnMesureLIst'

    response_avg = requests.post(url_avg, param_avg, verify=False, allow_redirects=True, timeout=90)
    avg_json = response_avg.json()
    avg_json = avg_json['list']
    import datetime
    dust_data = pd.DataFrame(columns=['Time', 'Dust'])
    for item in reversed(avg_json):
        ### because API use 24h (which is not supported by datetime. so, we convert it to 0h
        if '24:00' in item['dataTime']:
            item['dataTime'] = item['dataTime'].replace('24:00', '00:00')
            data_time = pd.to_datetime(item['dataTime'])
            original_date = data_time.date()  ## get date
            next_date = original_date + datetime.timedelta(days=1)  ### move date to previous
            item['dataTime'] = item['dataTime'].replace(str(original_date), str(next_date))

        dust_data = dust_data.append(pd.DataFrame([[item['dataTime'], item['gyeonggi']]], columns=['Time', 'Dust']),
                                     ignore_index=True)
    dust_data['Time'] = pd.to_datetime(dust_data['Time'])
    dust_data.set_index('Time', inplace=True)
    return dust_data


def get_dust_data_from_database(config, start_time, end_time):
    db = connect_to_data_base(config)
    cursor = db.cursor()
    query = 'select * from nisbcp.l_weather_pm where pmdt between "' + start_time + '" and "' + end_time + '"'
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        rows = [list(row) for row in data]  ## read row from query
        ## create dataframe and put data into
        dust_data = pd.DataFrame(rows, columns=['Time', 'Dust'], dtype=float)
        ## set index
        dust_data['Time'] = pd.to_datetime(dust_data['Time'])
        dust_data.set_index('Time', inplace=True)
        ## close database connection
        cursor.close()
        db.close()
    except:
        cursor.close()
        cursor.close()
        db.close()
        raise Exception('Could not execute the query, check statement or connection to database..')
    return dust_data


def reshapeForLSTM(data, time_steps=None):
    """
	:param data: intput data
	:param time_steps: time steps after
	:return: reshaped data for LSTM
	"""
    """
	The LSTM network expects the input data (X) 
	to be provided with 
	a specific array structure in the form of: 
	[samples, time steps, features].
	"""
    if time_steps is None:
        print("please denote 'time_steps'...!")
        return (None)
    else:
        data_reshaped = np.reshape(data, (data.shape[0], time_steps, 1))
    return (data_reshaped)







# --- create dataset with window size --- #
def sequentialize_1_feature_data(scaled_inputData, inputData_index, output_dim):
    '''
    transform 1 feature dataset to window_size dataset by shifting time steps
    :param scaled_inputData: scaled 1 dimension data
    :param inputData_index: input data's index (to convert to dataframe before shifting
    :param input_features: size of input data (input dimenssion)
    :param output_features: size of output data (output dimenssion)
    :return: reshaped input and output sequence
    '''
    assert isinstance(output_dim, int), 'output feature must be integer'

    # change type to use 'shift' of pd.DataFrame
    scaled_inputData = pd.DataFrame(scaled_inputData, index=inputData_index)

    # dataframe which is shifted as many as output_features
    for idx in range(1, output_dim + 1):
        scaled_inputData["column_{}".format(idx)] = scaled_inputData[scaled_inputData.columns[0]].shift(idx)
    # drop na
    inputSequence = scaled_inputData.dropna().drop(scaled_inputData.columns[0], axis=1)
    output = scaled_inputData.dropna()[[scaled_inputData.columns[0]]]

    ## convert to numpy array
    inputSequence = inputSequence.values
    output = output.values
    return np.reshape(inputSequence,(inputSequence.shape[0],output_dim,1)), output

# --- create dataset with window size --- #
# def sequentialize_multiple_features_data(scaled_inputData, inputData_index, output_indexes, config):
#     '''
#     transform 1 feature dataset to window_size dataset by shifting time steps
#     :param scaled_inputData: scaled 1 dimension data
#     :param inputData_index: input data's index (to convert to dataframe before shifting
#     :param input_features: size of input data (input dimenssion)
#     :param output_features: size of output data (output dimenssion)
#     :return: reshaped input and output sequence
#     '''
#     assert isinstance(output_indexes, list), 'Please indicate output features\' indexes'
#
#     # change type to use 'shift' of pd.DataFrame
#     scaled_inputData = pd.DataFrame(scaled_inputData, index=inputData_index)
#     ### count input data's columns (as input features)
#     input_dim = len(scaled_inputData.columns)
#
#     assert len(output_indexes) <= input_dim, 'output features cannot greater than input features'
#
#     # dataframe which is shifted as many as output_features
#     for idx in range(1, config.time_steps + 1):
#         for i in range(input_dim):
#             ### shift each column
#             scaled_inputData["column_{}_{}".format(i,idx)] = scaled_inputData[scaled_inputData.columns[i]].shift(idx)
#
#     # drop na
#     inputSequence = scaled_inputData.dropna().drop([scaled_inputData.columns[-j] for j in range(input_dim)], axis=1) ### drop output column and 2 last column [0,-2,-1]
#
#     output = scaled_inputData.dropna()[[scaled_inputData.columns[k] for k in output_indexes]] ## output is output index columns [0]
#
#     ## convert to numpy array
#     inputSequence = inputSequence.values
#     inputSequence = np.reshape(inputSequence,(inputSequence.shape[0],config.time_steps,input_dim))
#     output = output.values
#
#     # ### neeed more checking, do we need flipping?
#     # inputSequence = np.flip(inputSequence, axis=1)
#     # output = np.flip(output, axis=1)
#
#
#     return (inputSequence, output)


def sequentialize_multiple_features_data(scaled_inputData, inputData_index, output_indexes, config):
    '''
    transform 1 feature dataset to window_size dataset by shifting time steps
    :param scaled_inputData: scaled 1 dimension data
    :param inputData_index: input data's index (to convert to dataframe before shifting
    :param input_features: size of input data (input dimenssion)
    :param output_features: size of output data (output dimenssion)
    :return: reshaped input and output sequence
    '''
    assert isinstance(output_indexes, list), 'Please indicate output features\' indexes'
    scaled_inputData_ = scaled_inputData
    # change type to use 'shift' of pd.DataFrame
    scaled_inputData_ = pd.DataFrame(scaled_inputData_, index=inputData_index)
    ### count input data's columns (as input features)
    input_dim = len(scaled_inputData_.columns)

    assert len(output_indexes) <= input_dim, 'output features cannot greater than input features'

    # dataframe which is shifted as many as output_features
    for idx in range(1, config.time_steps + 1):
        for i in range(input_dim):
            ### shift each column
            scaled_inputData_["column_{}_{}".format(i,idx)] = scaled_inputData_[scaled_inputData_.columns[i]].shift(idx)

    # drop na
    inputSequence = scaled_inputData_.dropna().drop([scaled_inputData_.columns[j] for j in range(input_dim)], axis=1)

    output = scaled_inputData_.dropna()[[scaled_inputData_.columns[k] for k in output_indexes]]

    ## convert to numpy array
    inputSequence = inputSequence.values
    output = output.values

    return (np.reshape(inputSequence,(inputSequence.shape[0],config.time_steps,input_dim)), output)



def get_hourly_temp_by_date(date):
    import requests
    import urllib.parse as urlparser
    import datetime

    date = pd.to_datetime(date).date()
    today = datetime.date.today()
    if date == today - datetime.timedelta(days=1):
        next_date = pd.to_datetime(date)
        next_date = str(next_date.date()).replace('-', '').strip()
    else:
        next_date = pd.to_datetime(date) + datetime.timedelta(days=1)
        next_date = str(next_date.date()).replace('-', '').strip()
    date = str(date).replace('-', '').strip()
    url = 'https://data.kma.go.kr/apiData/getData'
    params = dict(
        type='json',
        dataCd='ASOS',
        dateCd='HR',
        startDt=date,
        startHh='00',
        endDt=next_date,
        endHh='23',
        stnIds='119',
        schListCnt='24',
        pageIndex='1',
        apiKey='3zrCi0JNWnE%2BmdIkxKiH/FVTXyU4aYXRinkR21ktFNfhPh9cZtyBbMJhnJjqwNjv',
    )
    params_ = urlparser.unquote(urlparser.urlencode(params,
                                                    doseq=True))  ## join params into url style (unquote to decode ASCII chars back to normal chars)
    url_ = url + '?' + params_  ### generate full url (cannot get data from separate url and params, so we join them)
    response = requests.get(url_, verify=False, allow_redirects=True, timeout=90)
    json = response.json()
    data = json[3]['info']
    ### create list  to save hourly temperature and time index
    temps = []
    times = []
    for item in data:
        try:
            temps.append(item['TA'])  ### get temperature of each hour from 'TA' variable
            times.append(pd.to_datetime(item['TM']))  ### get time index from 'TM' variable
        except:
            ### if data is missing, pass
            pass
    ### convert data to dataframe before returning
    temps = pd.DataFrame(temps, columns=['Temperature'], index=times)
    return temps  # int(round(float(np.mean(temps)),0))###


def add_hourly_temp_column(data):
    import time
    import copy
    freq = str(data.index.freq)
    ## make a copy of original data
    data_ = copy.deepcopy(data)

    ## because the API only support 1 hour frequency temperature, transform  15 minutes frequency to 1 hour frequency
    if freq == '<15 * Minutes>':
        data_ = data_.resample('H').sum()
        ## extract date
        dates = [str(date.date()) for date in data_.index]
        dates = sorted(set(dates))
        ### create empty dataframe to save temperature data
        temps_df = pd.DataFrame(columns=['Temperature'])
        ### for counting down  variable
        i = len(dates)
        for date in dates:
            #print('Adding hourly temperature from API. Remaining days: {}'.format(i))  ## print progress for controling
            i -= 1
            time.sleep(1)  ### kma's API does not allow getting data too fast.
            ### get hourly temperature of each date
            hourly_temp = get_hourly_temp_by_date(date)
            ## add to total dataframe
            temps_df = pd.concat([temps_df, hourly_temp], axis=0)
        ### add temperature data as a column to original dataframe
        new_data = pd.concat([data_, temps_df], axis=1)

        ### reindex back to 15 minutes frequency
        new_data_ = new_data.reindex(pd.date_range(new_data.index[0], data.index[-1], freq='15T')).resample(
            '15T').interpolate().dropna()
        ### cut out redundant parts
        new_data_ = new_data_.loc[data.index]
        ## append temperature column to original data (we do not use interpolated data because we want to keep data as close to reality as possible)
        new_data__ = pd.concat([data, new_data_['Temperature']], axis=1)
        return new_data__
    else:
        ## extract dates
        dates = [str(date.date()) for date in data.index]
        dates = sorted(set(dates))
        ### create empty dataframe to save temperature data
        temps_df = pd.DataFrame(columns=['Temperature'])
        ### for counting down  variable
        i = len(dates)
        for date in dates:
            i -= 1
            time.sleep(1)  ### kma's API does not allow getting data too fast.
            ### get hourly temperature of each date
            hourly_temp = get_hourly_temp_by_date(date)
            ## add to total dataframe
            temps_df = pd.concat([temps_df, hourly_temp], axis=0)
        ### add temperature data as a column to original dataframe
        new_data = pd.concat([data, temps_df], axis=1)
        return new_data


def add_hourly_temp_column_from_csv(data):
    ## get data frequency
    freq = str(data.index.freq)
    ### get temperature info from file
    temperatures = pd.read_csv('temperature.csv')
    ## convert index column to datetime and setindex
    temperatures[temperatures.columns[0]] = pd.to_datetime(temperatures[temperatures.columns[0]])
    temperatures.set_index(temperatures.columns[0], inplace=True)

    if freq == '<15 * Minutes>':
        ## transform temperature data to 15 minutes freq using interpolate
        temperatures = temperatures.resample('15T').interpolate()

    ### append temperature to data, keep only temperature that match with data's time
    data_with_temperature = data.join(temperatures,how='left')
    return data_with_temperature


def get_today_temperature():
    import requests
    param_avg = dict(
        serviceKey='a2khNs3hR3z2Xx4gSeXCI0qA2dhIKkMOtVhLLQGERAT9aB4j3bHUqXPII7MW+493KZ50nPr164PbSpj5LayUlg==',
        numOfRows='10',
        pageNo='1',
        _type='json',
        regId='11B20601',
        pageSize='10',
    )
    url_avg = 'http://newsky2.kma.go.kr/service/VilageFrcstDspthDocInfoService/WidOverlandForecast'
    response_avg = requests.get(url_avg, param_avg, verify=False, allow_redirects=True, timeout=90)
    avg_json = response_avg.json()
    avg_json_ = avg_json['response']['body']['items']['item']
    today_noon_temp = 0
    today_night_temp = 0
    for item in avg_json_:
        if item['numEf'] == 1:
            today_noon_temp = item['ta']
        if item['numEf'] == 2:
            today_night_temp = item['ta']

    today_temp = int((today_noon_temp + today_night_temp) / 2)
    return today_temp


def get_forecasted_temperature(date):
    def _get_html_source_code(url):
        _html = ''
        resp = requests.get(url)
        if resp.status_code == 200:
            _html = resp.text
        return _html

    from bs4 import BeautifulSoup
    import requests
    import pandas as pd
    url = 'https://www.wunderground.com/hourly/kr/suwon/date/{}?cm_ven=localwx_hour'.format(date)
    html_source_code = _get_html_source_code(url)

    soup = BeautifulSoup(html_source_code, 'html.parser')
    forecast_table = soup.find(id='hourly-forecast-table')
    table_body = forecast_table.find_all('tbody')
    table_rows = table_body[0].find_all('tr')

    temperature = []
    time = []
    for row in table_rows:
        cells = row.find_all('td')
        time_ = cells[0].find('span')
        temp_ = cells[2].find('span')

        temp_C = (float(temp_.get_text().replace('°F',''))-32) * (5/9) ## convert fahrenheit to celsius
        temp_C = round(temp_C,2)
        temperature.append(temp_C)
        time.append(time_.get_text())
    for idx, t in enumerate(time):
        time[idx] = pd.to_datetime(date +' '+ t)
    temp_df = pd.DataFrame({'Time':time, 'Temperature':temperature})
    temp_df.set_index('Time', inplace=True)
    return temp_df

def update_temperature_from_kma_API_to_csv():
    '''
    update temperature in past until yesterday. (to use latter instead of using kma's API directly)
    '''
    import time
    import datetime
    try:
        old_temp = pd.read_csv('temperature.csv')
        old_temp[old_temp.columns[0]] = pd.to_datetime(old_temp[old_temp.columns[0]])
        old_temp.set_index(old_temp.columns[0], inplace=True)
        start_time = old_temp.index[-1] + datetime.timedelta(hours=1)
    except FileNotFoundError as e:
        print(e)
        old_temp = pd.DataFrame(columns=['Temperature'])
        start_time = pd.to_datetime('2018-10-01')


    _, end_time = get_time_by_keyword('yesterday')

    dates = pd.date_range(start_time, end_time, freq='H')
    dates = sorted(set(dates.date))

    ### create empty dataframe to save temperature data
    temps_df = pd.DataFrame(columns=['Temperature'])

    for date in dates:
        time.sleep(1)  ### kma's API does not allow getting data too fast.
        ### get hourly temperature of each date
        try:
            hourly_temp = get_hourly_temp_by_date(date)
            ## add to total dataframe
            temps_df = pd.concat([temps_df, hourly_temp], axis=0)
        except Exception as e:
            print (e)
            pass
    new_temp = old_temp.combine_first(temps_df)
    today, _ = get_time_by_keyword('today')
    ### get forecasted temperature of next 7 day
    end_time = today + datetime.timedelta(days=7)
    dates = pd.date_range(start_time, end_time, freq='D')
    dates = sorted(set(dates.date))
    forecasted_temp = pd.DataFrame(columns=['Temperature'])

    for date in dates:
        oneday_forecasted_temp = get_forecasted_temperature(str(date))
        forecasted_temp = pd.concat([forecasted_temp, oneday_forecasted_temp], axis=0)
    new_temp = new_temp.combine_first(forecasted_temp)


    ### update weather data from ntels server
    try:
        new_temp.to_csv('temperature.csv')
        print('Updated temperature..')
    except PermissionError as e:
        raise PermissionError('Could not write temperature file, check permission..')




def get_predicted_consumption_from_API(input, building_id):
    import requests
    import pandas as pd
    # building_id='B0002'
    prediction_API_url = 'http://210.219.151.163:1280/prediction'
    headers = {
        'Content-Type': 'application/json',
        'cache-control': 'no-cache',
        'Postman-Token': '94ea6528-06d7-46e7-a7c2-5a38c198d1f8'
    }
    input_ = str(input).replace('\'', r'"').replace(' ', '')
    params = "{\"bld_id\" : \"" + building_id + "\",\"token\": \"D6099EC547FAC794B34542A82B12A12C586A3351A2892414CB175179787A894B\",\"data\" : " + input_ + ",\"timestamp\" : \"2018-11-21 09:42:46.502277\"}"
    try:
        response = requests.post(prediction_API_url, data=params, headers=headers, timeout=90)
    except:
        raise Exception('Could not connect to API server, check server configuration..'.format(prediction_API_url))
    try:
        result = response.json()['prediction_result']
        result_df = pd.DataFrame(result, columns=['Time', 'Power'])
        result_df.set_index('Time', inplace=True)
    except:
        result_df = pd.DataFrame(columns=['Power'])
    return result_df


