select  count(*) from comments as c,          votes as v,  		badges as b,  		users as u where u.Id  = c.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND b.Date<='2014-08-07 00:04:54'::timestamp  AND c.CreationDate<='2014-09-10 09:27:15'::timestamp  AND u.Reputation=101  AND u.UpVotes>=0  AND u.UpVotes<=39  AND u.CreationDate<='2014-09-05 15:47:22'::timestamp  AND v.VoteTypeId=2  AND v.CreationDate>='2010-08-19 00:00:00'::timestamp  AND v.CreationDate<='2014-09-08 00:00:00'::timestamp;