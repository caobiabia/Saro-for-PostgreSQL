select  count(*) from comments as c,          votes as v,  		badges as b,  		users as u where u.Id  = c.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND c.Score=0  AND u.Views>=0  AND u.CreationDate>='2010-07-20 04:27:02'::timestamp  AND u.CreationDate<='2014-09-04 07:09:25'::timestamp  AND v.BountyAmount<=50  AND v.CreationDate<='2014-09-09 00:00:00'::timestamp;