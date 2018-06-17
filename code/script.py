import pandas as pd
import numpy as np

uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('../data/uid_test_a.txt',index=None)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)

voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()

voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()

voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)

voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

voice['start_time']=voice['start_time'].astype(np.int32)
voice['end_time']=voice['end_time'].astype(np.int32)
voice['duration_time']=voice['end_time']-voice['start_time']
voice_average_duration_time=voice.groupby(['uid'])['duration_time'].mean().reset_index()
voice_average_duration_time.columns=['uid','average_duration_time']

voice['start_time']=voice['start_time'].astype(str)
voice['end_time']=voice['end_time'].astype(str)
voice['start_hour']=[int(x[2:4]) for x in voice['start_time']]
voice_average_start_hour=voice.groupby(['uid'])['start_hour'].mean().reset_index()
voice_average_start_hour.colums=['uid','voice_avarage_start_hour']

average_frequency_per_num=voice.groupby(['uid','opp_num'])['uid'].count().groupby(['uid']).mean()
voice_average_frequency_per_num=pd.DataFrame({'uid':average_frequency_per_num.index,'voice_average_frequency_per_num':average_frequency_per_num.values})

#voice['start_time']=voice['start_time'].astype(np.int32)
#voice['end_time']=voice['end_time'].astype(np.int32)
#voice.groupby(['uid'])['start_time'].agg({'average_interval' : lambda x: (x.max()-x.min())/len(arr)}).reset_index();

voice_in_out['voice_out_rate']=voice_in_out['voice_in_out_0']/(voice_in_out['voice_in_out_0']+voice_in_out['voice_in_out_1'])

voice_opp_len['voice_opp_long_len_ratio']=voice_opp_len.iloc[:,9:].sum(axis=1)/voice_opp_len.iloc[:,1:].sum(axis=1)

voice_call_type['voice_long_distance_call_ratio'] = voice_call_type.iloc[:,3:].sum(axis=1)/voice_call_type.iloc[:,1:].sum(axis=1)

voice_start_hour_static=voice.groupby(['uid'])['start_hour'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_start_hour').reset_index()




sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()

sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()

sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)


sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)


sms['start_hour']=[int(x[2:4]) for x in sms['start_time']]
sms_average_start_hour=sms.groupby(['uid'])['start_hour'].mean().reset_index()
sms_average_start_hour.columns=['uid','sms_average_start_hour'];

sms_start_hour_static=sms.groupby(['uid'])['start_hour'].agg(['std','max','min','median','mean','sum']).add_prefix('sms_start_hour').reset_index()



sms_average_frequency_per_num=sms.groupby(['uid','opp_num'])['uid'].count().groupby(['uid']).mean()
sms_average_frequency_per_num=pd.DataFrame({'uid':sms_average_frequency_per_num.index,'sms_average_frequency_per_num':sms_average_frequency_per_num})

sms_in_out['sms_out_rate']=sms_in_out['sms_in_out_0']/(sms_in_out['sms_in_out_0']+sms_in_out['sms_in_out_1'])

sms_opp_len['sms_opp_long_len_ratio']=sms_opp_len.iloc[:,12:].sum(axis=1)/sms_opp_len.iloc[:,1:].sum(axis=1)

sms['start_date']=[int(x[0:2]) for x in sms['start_time']]
sms_counts_ervery_day=sms.groupby(['uid','start_date'])['uid'].count().unstack().fillna(0)


wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()

visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()


up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()

down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()

wa_name['average_frequency_per_wa']=wa_name['wa_name_count']/wa_name['wa_name_unique_count']


wa_counts_every_day=wa.groupby(['uid','date'])['uid'].count().unstack().fillna(0)
wa_counts_every_day.columns=np.arange(1,46)
wa_sms_every_day_count_multiply=(wa_counts_every_day*sms_counts_ervery_day).fillna(0)



wa_sms_count_product_average= (wa_sms_every_day_count_multiply.sum(axis=1)/(wa_sms_every_day_count_multiply!=0).sum(axis=1))
wa_sms_count_product_average=wa_sms_count_product_average.fillna(wa_sms_count_product_average.mean())
wa_sms_count_product_average=pd.DataFrame({'uid':wa_sms_count_product_average.index,'wa_sms_count_product_average':wa_sms_count_product_average.values}).fillna(0)

wa_counts_every_day_stastics = wa_counts_every_day.max(axis=1).reset_index()
wa_counts_every_day_stastics.columns=['uid','wa_counts_every_day_max'];
wa_counts_every_day_stastics['wa_counts_every_day_std']=wa_counts_every_day.std(axis=1).values
wa_counts_every_day_stastics['wa_counts_every_day_median']=wa_counts_every_day.median(axis=1).values
wa_counts_every_day_stastics['wa_counts_every_day_mean']=wa_counts_every_day.mean(axis=1).values
wa_counts_every_day_stastics['wa_counts_every_day_sum']=wa_counts_every_day.sum(axis=1).values



wa_taobao=wa[wa['wa_name']=="淘宝网"]
visit_taobao_cnt = wa_taobao.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('taobao_visit_cnt_').reset_index()

wa_jingdong=wa[wa['wa_name']=="京东"]
visit_jingdong_cnt = wa_jingdong.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('jingdong_visit_cnt_').reset_index()

feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out, voice_average_duration_time,voice_average_start_hour,voice_average_frequency_per_num,
sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out, sms_average_start_hour,sms_average_frequency_per_num,
wa_name,visit_cnt,visit_dura,up_flow,down_flow,wa_sms_count_product_average, sms_start_hour_static,wa_counts_every_day_stastics]

train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')

test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')

train_feature.to_csv('../data/train_feature_1.0.csv',index=None)
test_feature.to_csv('../data/test_feature_1.0.csv',index=None)