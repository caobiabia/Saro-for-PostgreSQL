select  count(*) from comments as c,  		postHistory as ph,  		badges as b,          votes as v,          users as u  where u.Id =b.UserId 	and b.UserId = ph.UserId 	and ph.UserId = v.UserId 	and v.UserId = c.UserId  AND b.Date<='2014-09-11 12:04:01'::timestamp  AND ph.PostHistoryTypeId=25  AND u.Reputation>=1  AND u.Reputation<=131  AND u.Views<=56  AND v.VoteTypeId=2;