select  count(*) from postHistory as ph,          votes as v,  		users as u,  		badges as b  where u.Id = ph.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND b.Date<='2014-09-03 13:39:02'::timestamp  AND ph.PostHistoryTypeId=1  AND ph.CreationDate>='2010-10-26 13:52:10'::timestamp  AND ph.CreationDate<='2014-08-05 02:36:17'::timestamp  AND v.CreationDate>='2010-07-20 00:00:00'::timestamp  AND v.CreationDate<='2014-09-08 00:00:00'::timestamp;