select  count(*) from comments as c,  		postHistory as ph,  		badges as b,          votes as v,          users as u  where u.Id =b.UserId 	and b.UserId = ph.UserId 	and ph.UserId = v.UserId 	and v.UserId = c.UserId  AND c.CreationDate>='2010-07-22 11:38:20'::timestamp  AND c.CreationDate<='2014-09-07 21:39:29'::timestamp  AND ph.PostHistoryTypeId=24  AND ph.CreationDate<='2014-09-11 17:00:14'::timestamp  AND u.Reputation>=1  AND u.Reputation<=1403  AND u.Views>=0  AND v.VoteTypeId=2  AND v.CreationDate>='2010-07-19 00:00:00'::timestamp;